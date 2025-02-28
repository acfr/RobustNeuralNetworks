import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
from typing import Sequence
from flax.struct import dataclass

import pickle
from pathlib import Path
dirpath = Path(__file__).resolve().parent


# Can we update learning rate on the fly?
# Answer is yes!! https://github.com/google/flax/issues/4531

@dataclass
class TestParams:
    w: jax.Array
    b: jax.Array

class Linear(nnx.Module):
    def __init__(self, nu: int, ny: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        w = nnx.Param(jax.random.uniform(key, (nu, ny)))
        b = nnx.Param(jnp.zeros((ny,)))
        self.nu = nu
        self.ny = ny
        self.direct = TestParams(w, b)
        
        d = min(nu, ny)
        self.testing_nonjit = int(np.sum(np.linalg.eigvals(w[:d, :d])))

    def __call__(self, x: jax.Array):
        ps = self.direct
        return x @ ps.w + ps.b

class MLP(nnx.Module):
    def __init__(self, nu: int, nh: Sequence[int], ny, *, rngs: nnx.Rngs):
        self.nu = nu
        self.nh = nh
        self.ny = ny
        self.activation = nnx.relu
        
        in_sizes = (nu,) + nh
        out_sizes = nh + (ny,)
        self.layers = [
            Linear(in_sizes[k], out_sizes[k], rngs=rngs)
            for k in range(len(in_sizes))
        ]
    
    def __call__(self, x: jax.Array):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


# Dummy data
seed = 0
batches = 20
nu, ny = 2, 5
nh = (2, 4, 8)
inputs = jnp.ones((batches, nu))
y_out = jnp.ones((batches, ny))

# Model
model = MLP(nu, nh, ny, rngs=nnx.Rngs(params=seed))
outputs = model(inputs)

# Create an optimiser with a learning rate scheduler
optimiser = optax.chain(
    optax.adam(1e-3),
    optax.contrib.reduce_on_plateau(
        patience=1,
        factor=0.1,
    )
)
optimiser = nnx.Optimizer(model, optimiser)

# Do some training
def loss_fn(model: MLP, x, y):
    y_pred = model(x)
    return jnp.mean((y - y_pred)**2), y_pred

grad_loss = nnx.value_and_grad(loss_fn, has_aux=True)

@nnx.jit # NOTE: can't use jax.jit for the model anymore.
def train_step(model, optimiser, x, y):
    (loss, _), grads = grad_loss(model, x, y)
    # Could transform the grads here...?
    # grads = jax.tree.map(lambda g: 0*g, grads)
    optimiser.update(grads, value=loss) # Second argument for the schedule
    return loss

for _ in range(10):
    loss = train_step(model, optimiser, inputs, y_out)
    

print(f'{loss = }')
print(f'{optimiser.step.value = }')

# Can only use jax.jit normally if we first split into graphdef and state
# Hopefully we won't have to though
@jax.jit
def forward(graphdef: nnx.GraphDef, state: nnx.State, x):
    model = nnx.merge(graphdef, state)
    return model(x)

graphdef, state = nnx.split(model)
out = forward(graphdef, state, inputs)
# print(f'{out = }')

# Let's try using model inside another jitted function
@jax.jit
def rollout(carry, unused_t):
    u, = carry
    y = model(u)
    unext = u * y[1,1]
    return (unext,), y

_, out = jax.lax.scan(rollout, (inputs,), length=3)
print(out[-1])

# Do file saving by only saving the model state
_, state = nnx.split(model)
data = ({"test": 0}, state, out)

filepath = Path("test.pickle")
with filepath.open('wb') as fout:
    pickle.dump(data, fout)
    
with filepath.open('rb') as fin:
        buf = fin.read()
data = pickle.loads(buf)
old_state = data[1]

# Create dummy model and fill in the state
new_model = MLP(nu, nh, ny, rngs=nnx.Rngs(params=seed))
graphdef, _ = nnx.split(new_model)
new_model = nnx.merge(graphdef, old_state)

_, out = jax.lax.scan(rollout, (inputs,), length=3)
print(out[-1])
