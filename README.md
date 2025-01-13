# Prodigy + ScheduleFree
*Eliminating hyperparameters, one commit at a time.*

**Current status:** Experimental

## Installation
```
pip install prodigy-plus-schedule-free
```

## Usage
```python
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
optimizer = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, betas=(0.9, 0.99), beta3=None, 
                                    weight_decay=0.0, weight_decay_by_lr=True, 
				    use_bias_correction=False, d0=1e-6, d_coef=1.0, 
				    prodigy_steps=0, use_speed=False, eps=1e-8, 
				    split_groups=True, split_groups_mean=True,
 				    factored=True, fused_back_pass=False, use_stableadamw=True,
 				    use_muon_pp=False, use_cautious=False, use_grams=False,
                                    use_adopt=False, stochastic_rounding=True)
```

As with the reference implementation of schedule-free, a constant scheduler should be used, along with the appropriate
calls to `optimizer.train()` and `optimizer.eval()`. See the schedule-free documentation for more details: https://github.com/facebookresearch/schedule_free

## Details
An optimiser based on Prodigy that includes schedule-free logic and much, much lower memory usage, the aim being to remove the need to set any hyperparameters. Of course,
that's never the case with any optimiser, but hopefully, this comes close!

Hyperparameters eliminated: Learning rate (Prodigy), LR scheduler (ScheduleFree), epsilon (Adam-atan2, optional, not enabled by default).

Based on code from:
* https://github.com/facebookresearch/schedule_free
* https://github.com/konstmish/prodigy

Incorporates improvements from these pull requests (credit to https://github.com/dxqbYD, https://github.com/sangoi-exe and https://github.com/nhamanasu):
* https://github.com/konstmish/prodigy/pull/23
* https://github.com/konstmish/prodigy/pull/22
* https://github.com/konstmish/prodigy/pull/20
* https://github.com/facebookresearch/schedule_free/pull/54

If you do use another scheduler, linear or cosine is preferred, as a restarting scheduler can confuse Prodigy's adaptation logic.

Leave `lr` set to 1 unless you encounter instability. Do not use with gradient clipping, as this can hamper the
ability for the optimiser to predict stepsizes. Gradient clipping/normalisation is already handled in the following configurations:

1) `use_stableadamw=True,eps=1e8` (or any reasonable positive epsilon. This is the default.)
2) `eps=None` (Adam-atan2, scale invariant. Will disable StableAdamW if enabled.)

By default, `split_groups` and `split_groups_mean` are set to `True`, so each parameter group will have its own `d` values, however,
they will all use the harmonic mean for the dynamic learning rate. To make each group use its own dynamic LR, set `split_groups_mean` to False.
To use the reference Prodigy behaviour where all groups are combined, set `split_groups` to False. 

The optimiser uses low-rank approximations for the second moment, much like Adafactor. There should be little to no difference 
in training performance, but your mileage may vary. If you encounter problems, you can try disabling factorisation by 
setting `factored` to `False`.

The optimiser also supports [fused backward pass](https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html) to significantly lower
gradient memory usage. The `fused_back_pass` argument must be set to `True` so the optimiser knows not to perform the regular step. Please note however that 
your training scripts / UI of choice *must* support the feature for generic optimisers -- as of January 2025, popular trainers such as OneTrainer and Kohya 
hard-code which optimisers have fused backward pass support, and so this optimiser's fused pass will not work out of the box with them.

In some scenarios, it can be advantageous to freeze Prodigy's adaptive stepsize after a certain number of steps. This
can be controlled via the `prodigy_steps` settings. [It's been suggested that all Prodigy needs to do is achieve "escape velocity"](https://arxiv.org/pdf/2409.20325)
in terms of finding a good LR, which it usually achieves after ~25% of training, though this is very dependent on batch size and epochs. 

This setting can be particularly helpful when training diffusion models, which have very different gradient behaviour than what most optimisers are tuned for. 
Prodigy in particular will increase the LR forever if it is not stopped or capped in some way (usually via a decaying LR scheduler).

## Experimental features

**Adam-atan2:** Enabled by setting `eps` to `None`. Outlined in [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872), 
you can use atan2 in place of the regular division plus epsilon found in most Adam-style optimisers. This makes updates scale-invariant, and removes the need 
to tweak the epsilon. Disabled by default.

**Muon:** Enabled by setting `use_muon_pp` to `True`. This changes the fundamental behaviour of the optimiser for compatible parameters from AdamW to SGD
with a quasi-second moment based on the RMS of the updates. [As explained by Keller Jordan](https://x.com/kellerjordan0/status/1844782418676339059), and demonstrated 
(in various forms) by optimisers such as Shampoo, SOAP and Jordan's Muon, applying preconditioning to the gradient can improve convergence. However, 
this approach may not work in some situations (small batch sizes, fine-tuning) and as such, is disabled by default.

**C-Optim:** Enabled by setting `use_cautious` to `True`. Outlined in [Cautious Optimizers: Improving Training with One Line of Code](https://arxiv.org/pdf/2411.16085). 
Applies a simple modification to parameter updates that promotes values that are aligned with the current gradient. This should result in faster convergence. While not 1:1 compatible with schedule-free, [the implementation by nhamanasu](https://github.com/facebookresearch/schedule_free/pull/54) does work, though improvements may be limited.

**Grams:** Enabled by setting `use_grams` to `True`. Described in [Grams: Gradient Descent with Adaptive Momentum Scaling](https://arxiv.org/abs/2412.17107). 
In a similar vein to C-Optim, the parameter update is modified to separate the update direction from momentum. Thanks to [gesen2egee for the pull request](https://github.com/LoganBooker/prodigy-plus-schedule-free/pull/5).

**ADOPT:** Enabled by setting `use_adopt` to `True`. A partial implementation of [ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate](https://arxiv.org/abs/2411.02853), as we only update the second moment after the parameter update, so as to exclude the current gradient. Disabled by default.

**OrthoGrad:** Enabled by setting `use_orthograd` to `True`. Updates weights using the component of the gradient that is orthogonal to the current weight direction, as described in [Grokking at the Edge of Numerical Stability](https://arxiv.org/pdf/2501.04697). Can help prevent overfitting and improve generalisation. Ignored
for parameters using Muon.

**SPEED:** Enabled by setting `use_speed` to `True`. Something of my own creation I've dubbed "Signed Prodigy with ExponEntial D", or SPEED. Prodigy is very
dependent on the magnitude of weights, updates and the gradient, which makes it very difficult to apply other types of optimisations to it. This is my attempt to
decouple Prodigy's LR adaptation from these magnitudes by using just the sign instead, along with a capped growth rate.

## MNIST results
Generated from the [MNIST example in the schedule-free repository](https://github.com/facebookresearch/schedule_free/tree/main/examples/mnist), using the default settings.
```
Prodigy LR: 0.000832
Test set: Average loss: 0.0472, Accuracy: 9836/10000 (98.36%)
Test set: Average loss: 0.0345, Accuracy: 9879/10000 (98.79%)
Test set: Average loss: 0.0305, Accuracy: 9905/10000 (99.05%)
Test set: Average loss: 0.0295, Accuracy: 9912/10000 (99.12%)
Test set: Average loss: 0.0296, Accuracy: 9916/10000 (99.16%)
Test set: Average loss: 0.0295, Accuracy: 9921/10000 (99.21%)
Test set: Average loss: 0.0305, Accuracy: 9916/10000 (99.16%)
Test set: Average loss: 0.0300, Accuracy: 9915/10000 (99.15%)
Test set: Average loss: 0.0305, Accuracy: 9917/10000 (99.17%)
Test set: Average loss: 0.0310, Accuracy: 9919/10000 (99.19%)
Test set: Average loss: 0.0326, Accuracy: 9923/10000 (99.23%)
Test set: Average loss: 0.0338, Accuracy: 9928/10000 (99.28%)
Test set: Average loss: 0.0345, Accuracy: 9925/10000 (99.25%)
Test set: Average loss: 0.0354, Accuracy: 9925/10000 (99.25%)
```

## Recommended usage

Earlier versions of the optimiser recommended setting `prodigy_steps` equal to 5-25% of your total step count, but this should not be necessary with recent updates. That said,
you can still use the setting to make sure the LR does not change after a certain step, and free any memory used by Prodigy for adapting the step size.
