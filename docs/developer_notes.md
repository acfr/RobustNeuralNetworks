# Developer Notes


## Initialising from Explicit Model

One cause for slighlty convoluted code is that anything in the `setup()` method must be jittable. This means that certain types of error checking are not possible (anything that would mess with the JAX tracer). It also means that when we want to initialise a REN from an explicit model, we need to have a separate `model.pre_init()` which handles all the non-jittable code.

If `setup()` includes any non-jittable code, then we cannot JIT `model.apply()`. It would be great to have a more flexible init setup for `flax` models that allows this.

## Solving these problems

`flax` has recently introduced `flax.nnx` as a replacement for `flax.linen`. It simplifies the construction of networks to be more pythonic, and it will solve this problem. Switch to it at some point. Have chosen not to for now just because most codebases use `flax.linen`.
