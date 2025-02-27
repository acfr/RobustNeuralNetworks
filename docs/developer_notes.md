# Developer Notes


## Splitting Direct and Explicit Params

The current code to split up the `model.direct_to_explicit()` and `model.explicit_call()` methods is a little bit convoluted. The reason is because anything that is created in `model.setup()` cannot be accessed outside of the `model.init()` and `model.apply()` methods for `flax.linen` modules.

This is particularly problematic for the `LBDN` and `ScalableREN` models, which both use other network models in their construction (`LBDN` uses the `SandwichLayer`, `ScalableREN` uses the `LBDN`). If we were to have something like this:

```python
class MyNetwork(nn.Module):
    def setup(self):
        self.network = LBDN(...)

    def __call__(self, x):
        return self.network(x)
```

then we can't split it into an explicit call with the following:

```python
    def explicit_call(self, x):
        return self.network(x)
```

because `network` was defined in the `setup()`. This is a little bit frustrating, and a lot of the code could be simplified if we could avoid this. Let's see how future iterations of `flax` evolve.

## Initialising from Explicit Model

Another cause for slighlty convoluted code is that anything in the `setup()` method must be jittable. This means that certain types of error checking are not possible (anything that would mess with the JAX tracer). It also means that when we want to initialise a REN from an explicit model, we need to have a separate `model.pre_init()` which handles all the non-jittable code.

If `setup()` includes any non-jittable code, then we cannot JIT `model.apply()`. Again, it would be great to have a more flexible init setup for `flax` models that allows this.

## Solving these problems

`flax` has recently introduced `flax.nnx` as a replacement for `flax.linen`. It is worth looking into this as a potential fix for the above. Does the new architecture address the limitations?

Beware though: `flax.nnx` uses stateful models (I think) which could cause a complete re-write to any code that uses the REN code. Potentially best leaving this out for the moment so that it's still compatible with existing codebases.
