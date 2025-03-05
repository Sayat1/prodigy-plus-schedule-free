import math
import torch
from statistics import harmonic_mean

class CoreOptimiser(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        if not 0.0 < kwargs['d0']:
            raise ValueError("Invalid d0 value: {}".format(kwargs['d0']))
        if not 0.0 < kwargs['lr']:
            raise ValueError("Invalid learning rate: {}".format(kwargs['lr']))
        if kwargs['eps'] is not None and not 0.0 < kwargs['eps']:
            raise ValueError("Invalid epsilon value: {}".format(kwargs['eps']))
        if not 0.0 <= kwargs['betas'][0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(kwargs['betas'][0]))
        if kwargs['betas'][1] is not None and not 0.0 <= kwargs['betas'][1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(kwargs['betas'][1]))
        if kwargs['beta3'] is not None and not 0.0 <= kwargs['beta3'] < 1.0:
            raise ValueError("Invalid beta3 parameter: {}".format(kwargs['beta3']))

        self.try_hook_kohya_fbp()

        if kwargs['eps'] is None:
            print(f"[{self.__class__.__name__}] 'eps' is None, Adam-atan2 enabled.")
            if kwargs['use_stableadamw']:
                print(f"[{self.__class__.__name__}] 'use_stableadamw' has been disabled (mutually exclusive with Adam-atan2).")
                kwargs['use_stableadamw'] = False

        if kwargs['use_cautious'] and kwargs['use_grams']:
            print(f"[{self.__class__.__name__}] 'use_grams' has been disabled (mutually exclusive with 'use_cautious').")
            kwargs['use_grams'] = False

        if kwargs['use_focus']:
            if kwargs['factored']:
                print(f"[{self.__class__.__name__}] 'factored' has been disabled (incompatible with 'use_focus').")
                kwargs['factored'] = False
            if kwargs['eps'] is None:
                print(f"[{self.__class__.__name__}] Adam-atan2 ('eps=None') has been disabled (incompatible with 'use_focus').")
                # We skip the Adam-atan2 branch entirely when FOCUS is enabled.

        split_groups = kwargs.pop('split_groups')
        split_groups_mean = kwargs.pop('split_groups_mean')
        fused_back_pass = kwargs.pop('fused_back_pass')

        defaults = dict(kwargs)
        
        defaults['d'] = defaults['d_prev'] = defaults['d0']
        defaults['weight_sum'] = defaults['d_denom'] = defaults['d_numerator'] = 0
        defaults['train_mode'] = True
        defaults['k'] = 1

        super().__init__(params, defaults)

        self.d0 = defaults['d0']
        if split_groups and len(self.param_groups) == 1:
            print(f"[{self.__class__.__name__}] Optimiser contains single param_group -- 'split_groups' has been disabled.")
            split_groups = False

        self.split_groups = split_groups
        self.split_groups_mean = split_groups_mean

        # Properties for fused backward pass.
        self.parameters_to_process = None
        self.shared_d = None
        self.fused_back_pass = fused_back_pass

        # Use tensors to keep everything on device during parameter loop.
        for group in (self.param_groups if self.split_groups else self.param_groups[:1]):
            p = group['params'][0]
            group['running_d_numerator'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
            group['running_d_denom'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def eval(self):
        pass

    @torch.no_grad()
    def train(self):
        pass

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True
    
    def supports_fused_back_pass(self):
        return True

    @torch.no_grad()
    def get_sliced_tensor(self, tensor, slice_p=11):
        return tensor.ravel()[::slice_p]

    @torch.no_grad()
    def check_running_values_for_group(self, p, group):
        if not self.split_groups:
            group = self.param_groups[0]

        if group['running_d_numerator'].device != p.device:
            group['running_d_numerator'] = group['running_d_numerator'].to(p.device)
        if group['running_d_denom'].device != p.device:
            group['running_d_denom'] = group['running_d_denom'].to(p.device)

    @torch.no_grad()
    def get_running_values_for_group(self, group):
        if not self.split_groups:
            group = self.param_groups[0]

        return group['running_d_numerator'], group['running_d_denom']

    @torch.no_grad()
    def get_d_mean(self):
        if self.split_groups and self.split_groups_mean:
            return harmonic_mean(group['d'] for group in self.param_groups)
        return None

    # Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
    def orthograd(self, p, grad):
        if p.norm(2) <= 1e-30:
            return grad.to(dtype=torch.float32, copy=True)

        G_shape = grad.shape
        w = p.view(-1)
        g = grad.view(-1)

        proj = torch.dot(w, g) / (torch.dot(w, w) + 1e-30)
        g_orth = g.to(dtype=torch.float32, copy=True).sub_(w, alpha=proj)
        g_orth_scaled = g_orth.mul_(g.norm(2) / (g_orth.norm(2) + 1e-30))

        return g_orth_scaled.view(G_shape)

    # Implementation by Nerogar. From: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    def copy_stochastic_(self, target, source):
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

    def smart_copy(self, target, source, stochastic_rounding, smart_delete_source):
        if target is source:
            return

        if stochastic_rounding and target.dtype == torch.bfloat16 and source.dtype == torch.float32:
            self.copy_stochastic_(target, source)
        else:
            target.copy_(source)

        if smart_delete_source:
            del source

    # Modified Adafactor factorisation implementation by Ross Wightman 
    # https://github.com/huggingface/pytorch-image-models/pull/2320
    @torch.no_grad()
    def factored_dims(self,
        shape,
        factored,
        min_dim_size_to_factor):
        r"""Whether to use a factored second moment estimator.
        This function returns a tuple with the two largest axes to reduce over.
        If all dimensions have size < min_dim_size_to_factor, return None.
        Args:
        shape: an input shape
        factored: whether to use factored second-moment estimator for > 2d vars.
        min_dim_size_to_factor: only factor accumulator if all array dimensions are greater than this size.
        Returns:
        None or a tuple of ints
        """
        if not factored or len(shape) < 2:
            return None
        if all(dim < min_dim_size_to_factor for dim in shape):
            return None
        sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
        return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])
    
    @torch.no_grad()
    def initialise_state(self, p, group):
        raise Exception("Not implemented!")

    @torch.no_grad()
    def initialise_state_internal(self, p, group):
        state = self.state[p]
        needs_init = len(state) == 0
        
        if needs_init:
            grad = p.grad
            dtype = torch.bfloat16 if grad.dtype == torch.float32 else grad.dtype
            sliced_data = self.get_sliced_tensor(p)

            if group['use_focus']:
                state['exp_avg_sq'] = torch.zeros_like(grad, memory_format=torch.preserve_format).detach()
            else:
                # NOTE: We don't initialise z/exp_avg here -- subclass needs to do that.
                factored_dims = self.factored_dims(
                    grad.shape,
                    factored=group['factored'],
                    min_dim_size_to_factor=32
                )

                if factored_dims is not None:
                    # Store reduction variables so we don't have to recalculate each step.
                    dc, dr = factored_dims
                    row_shape = list(grad.shape)
                    row_shape[dr] = 1
                    col_shape = list(grad.shape)
                    col_shape[dc] = 1
                    reduce_dc = dc - 1 if dc > dr else dc

                    factored_dtype = torch.float32 if group['factored_fp32'] else grad.dtype
                    state["exp_avg_sq"] = [torch.zeros(row_shape, dtype=factored_dtype, device=p.device).detach(), 
                                           torch.zeros(col_shape, dtype=factored_dtype, device=p.device).detach(), 
                                           dr, dc, reduce_dc]
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad, memory_format=torch.preserve_format).detach()

            # If the initial weights are zero, don't bother storing them.
            if p.any() > 0:
                state['p0'] = sliced_data.to(dtype=dtype, memory_format=torch.preserve_format, copy=True).detach()
            else:
                state['p0'] = torch.tensor(0.0, dtype=dtype, device=p.device)

            if not group['use_speed']:
                state['s'] = torch.zeros_like(sliced_data, memory_format=torch.preserve_format, dtype=dtype).detach()

        return state, needs_init

    def get_betas(self, group):
        beta1, beta2 = group['betas']
        if beta2 is None:
            beta2 = 1 - (1 / group['k'])
        return (beta1, beta2)

    def get_beta3(self, group):
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = self.get_betas(group)[1] ** 0.5
        return beta3

    @torch.no_grad()
    def update_d_stats_and_reset(self, group):
        k = group['k']
        prodigy_steps = group['prodigy_steps']
        
        if prodigy_steps > 0 and k >= prodigy_steps:
            return

        d, d0 = group['d'], group['d0']
        beta3 = self.get_beta3(group)

        running_d_numerator, running_d_denom = self.get_running_values_for_group(group)

        max_d_numerator = group.get('max_d_numerator', 0)
        d_numerator = group['d_numerator']
        d_numerator *= beta3

        d_numerator_item = running_d_numerator.item()
        d_denom_item = running_d_denom.item()

        # Prevent the accumulation of negative values in the numerator in early training.
        # We still allow negative updates once progress starts being made, as this is 
        # important for regulating the adaptive stepsize.
        if d > d0 or d_numerator_item > 0:
            d_numerator += d_numerator_item

        group['prev_d_numerator'] = group['d_numerator']
        group['max_d_numerator'] = max(d_numerator, max_d_numerator)
        group['d_numerator'] = d_numerator
        group['d_denom'] = d_denom_item

        running_d_numerator.zero_()
        running_d_denom.zero_()

    @torch.no_grad()
    def calculate_d(self, group):
        k = group['k']
        prodigy_steps = group['prodigy_steps']
        
        if prodigy_steps > 0 and k >= prodigy_steps:
            return

        d, d_coef = group['d'], group['d_coef']
        d_numerator, d_denom = group['d_numerator'], group['d_denom']

        if group['use_speed']:
            prev_d_numerator, max_d_numerator = group['prev_d_numerator'], group['max_d_numerator']

            if d_numerator >= max_d_numerator and prev_d_numerator > 0:
                d_hat = min(2 ** 0.5, (d_numerator / prev_d_numerator) ** 0.75)
                d = max(d, d * d_hat * d_coef)
        elif d_denom > 0:
            d_hat = (d_coef * d_numerator) / d_denom
            d = max(d, d_hat)

        group['d_prev'] = group['d']
        group['d'] = d

    def on_start_step(self, p, group):
        if self.parameters_to_process is None or self.parameters_to_process == 0:
            # Optimiser hasn't run yet (or is starting a new step), so initialise.
            self.parameters_to_process = sum(len(group['params']) for group in self.param_groups)
            # Check running values are on-device.
            if self.split_groups:
                for other_group in self.param_groups:
                    self.check_running_values_for_group(p, other_group)
            else:
                self.check_running_values_for_group(p, group)

    def on_end_step(self):
        self.parameters_to_process -= 1

        if self.parameters_to_process == 0:
            # Update d for next optimiser step.
            if self.split_groups:
                for i, group in enumerate(self.param_groups):
                    if group['prodigy_steps'] > 0 and group['k'] == group['prodigy_steps']:
                        print(f"[{self.__class__.__name__}] Prodigy stepsize adaptation disabled after {group['k']} steps for param_group {i}.")

                    self.update_d_stats_and_reset(group)

                for group in self.param_groups:
                    self.calculate_d(group)
                    group['weight_sum'] = group.get('running_weight_sum', 0)
                    group['k'] += 1

                self.shared_d = self.get_d_mean()
            else:
                # When groups aren't split, calculate d for the first group (which collects stats for all groups in non-split mode), 
                # then copy to all other groups.
                first_group = self.param_groups[0]
                self.update_d_stats_and_reset(first_group)
                self.calculate_d(first_group)
                
                for i, group in enumerate(self.param_groups):
                    if group['prodigy_steps'] > 0 and group['k'] == group['prodigy_steps']:
                        print(f"[{self.__class__.__name__}] Prodigy stepsize adaptation disabled after {group['k']} steps for param_group {i}.")

                    group['d'] = first_group['d']
                    group['d_numerator'] = first_group['d_numerator']
                    group['d_denom'] = first_group['d_denom']
                    group['weight_sum'] = group.get('running_weight_sum', 0)
                    group['k'] += 1


    def get_dlr(self, group):
        dlr = (self.shared_d if self.split_groups and self.shared_d else group['d']) * group['lr']
        return dlr * group.get('rect', 1.0)

    def update_prodigy(self, state, group, grad, data):
        k = group['k']
        prodigy_steps = group['prodigy_steps']
        
        if prodigy_steps <= 0 or k < prodigy_steps:
            beta3 = self.get_beta3(group)

            sliced_grad = self.get_sliced_tensor(grad)
            sliced_data = self.get_sliced_tensor(data)

            running_d_numerator, running_d_denom = self.get_running_values_for_group(group)

            x0_minus = state['p0'] - sliced_data
            x0_dot = torch.dot(sliced_grad, x0_minus)

            if group['use_speed']:
                d_update = group['d'] / (group['d0'] ** 0.5)
                x0_dot /= x0_minus.abs().sum().clamp_min(1e-8)
            else:
                d_update = group['d'] ** 0.5
                s = state['s']
                s.mul_(beta3).add_(sliced_grad, alpha=d_update)
                running_d_denom.add_(s.abs().sum())

            running_d_numerator.add_(x0_dot, alpha=d_update)

            del x0_minus
        else:
            # Free the memory used by Prodigy, as we no longer need it.
            if 's' in state:
                s = state.pop('s')
                del s
            if 'p0' in state:
                p0 = state.pop('p0')
                del p0

    def update_(self, num, denom, state, group, w):
        if group['use_focus']:
            # FOCUS: First Order Concentrated Updating Scheme: https://arxiv.org/pdf/2501.12243
            gamma = 0.1

            # Original form.
            # update = torch.sign(num) + gamma * torch.sign(w - denom)

            denom = denom.sub_(w).sign_().mul_(-gamma)
            update = num.sign_().add_(denom)
        else:
            eps = group['eps']

            if eps is None:
                # Approximate scaling for a regular Adam-style update.
                b = state.get('exp_clip_threshold', self.get_max_clip_threshold(group))

                # Adam-atan2. Use atan2 rather than epsilon and division 
                # for parameter updates (https://arxiv.org/abs/2407.05872).
                # Has the nice property of "clipping" the gradient as well.
                update = num.atan2_(denom.mul_(b)).mul_(b)
                self.compute_adaptive_rms(state, group, update)
            else:
                update = num.div_(denom.add_(eps))

        return update

    def get_denom(self, state, group):
        exp_avg_sq = state['exp_avg_sq']

         # Adam EMA updates
        if isinstance(exp_avg_sq, list):
            row_var, col_var, _, _, reduce_dc = exp_avg_sq

            row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True).add_(1e-30)
            denom = (row_var.div(row_col_mean) * col_var).sqrt_()
        elif group['use_focus']:
            denom = exp_avg_sq.clone()
        else:
            denom = exp_avg_sq.sqrt()

        return denom
   
    def update_first_moment(self, state, group, grad, beta1):
        exp_avg = state['exp_avg']
        d_k = group['d_prev'] / group['d']

        return exp_avg.mul_(beta1 * d_k).add_(grad, alpha=1 - beta1)

    def update_second_moment(self, state, group, grad, beta2, w, return_denom=True, denom_before_update=False):
        exp_avg_sq = state['exp_avg_sq']
        d_k = (group['d_prev'] / group['d']) ** 2

        denom = None

        if return_denom and denom_before_update:
            denom = self.get_denom(state, group)

        # Adam EMA updates
        if group['use_focus']:
            exp_avg_sq.mul_(beta2 * d_k).add_(w, alpha=1 - beta2)
        else:
            if isinstance(exp_avg_sq, list):
                row_var, col_var, dr, dc, _ = exp_avg_sq

                row_var.mul_(beta2 * d_k).add_(
                    grad.norm(dim=dr, keepdim=True).square_().mul_(1 / grad.shape[dr]),
                    alpha=1 - beta2
                )
                col_var.mul_(beta2 * d_k).add_(
                    grad.norm(dim=dc, keepdim=True).square_().mul_(1 / grad.shape[dc]),
                    alpha=1 - beta2
                )
            else:
                exp_avg_sq.mul_(beta2 * d_k).addcmul_(grad, grad, value=1 - beta2)

        if return_denom and denom is None:
            denom = self.get_denom(state, group)

        return denom

    def get_max_clip_threshold(self, group):
        _, beta2 = self.get_betas(group)
        # Maximum RMS of first update.
        return (1 - beta2) ** -0.5

    def compute_adaptive_rms(self, state, group, update):
        rms = self.get_rms(update, 1)

        if not group['adaptive_stableadamw']:
            return rms

        max_clip_threshold = state.get('exp_clip_threshold', self.get_max_clip_threshold(group))

        # Only adapt RMS once LR starts increasing.
        if group['d'] > group['d0'] and group['d'] == group['d_prev']:
            beta = 0.95
            max_clip_threshold = max_clip_threshold * beta + rms * (1 - beta)
            state['exp_clip_threshold'] = max_clip_threshold

        return max(rms / max_clip_threshold, 1)

    def get_rms(self, tensor, eps=1e-8):
        return tensor.norm(2).div(tensor.numel() ** 0.5).clamp_min(eps)

    def try_hook_kohya_fbp(self):
        self.kohya_original_patch_adafactor_fused = None

        try:
            # Import and patching will fail if not Kohya.
            import library.adafactor_fused

            # Get the original method so we can restore it later.
            self.kohya_original_patch_adafactor_fused = library.adafactor_fused.patch_adafactor_fused

            # Define the override.
            def prodigy_patch_adafactor_fused(optimizer):
                unwrapped_optimiser = None
                if hasattr(optimizer, "optimizer"):
                    # If the optimiser is wrapped, forward the calls to the actual optimiser.
                    def _step(self, *args, **kwargs):
                        return self.optimizer.step(*args, **kwargs)

                    def _step_param(self, *args, **kwargs):
                        return self.optimizer.step_param(*args, **kwargs)

                    optimizer.step = _step.__get__(optimizer)
                    optimizer.step_param = _step_param.__get__(optimizer)
                    unwrapped_optimiser = optimizer.optimizer
                else:
                    unwrapped_optimiser = optimizer
               
                print(f"[{self.__class__.__name__}] Kohya pipeline detected with fused backward pass. Gradient hook patch successful.")
                library.adafactor_fused.patch_adafactor_fused = unwrapped_optimiser.kohya_original_patch_adafactor_fused # Restore the original method.

                unwrapped_optimiser.fused_back_pass = True
                unwrapped_optimiser.kohya_original_patch_adafactor_fused = None

            # Patch the method.
            library.adafactor_fused.patch_adafactor_fused = prodigy_patch_adafactor_fused
        except:
            pass

    def try_unhook_kohya_fbp(self):
        if self.kohya_original_patch_adafactor_fused is None:
            return

        try:
            # Import and patching will fail if not Kohya.
            import library.adafactor_fused

            # User did not opt for fused backward pass, so remove our hook.
            library.adafactor_fused.patch_adafactor_fused = self.kohya_original_patch_adafactor_fused
        except:
            pass

        self.kohya_original_patch_adafactor_fused = None

    @torch.no_grad()
    def step_param(self, p, group):
        raise Exception("Not implemented!")            

    @torch.no_grad()
    def step_parameter(self, p, group, i):
        self.step_param(p, group)

    @torch.no_grad()
    def step(self, closure=None):
        self.try_unhook_kohya_fbp()

        if self.fused_back_pass:
            return
        
        """Performs a single optimisation step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            for p in param_group["params"]:
                self.step_param(p, param_group)

        return loss