import jax
import jax.numpy as jnp

from robustnn.ren_jax.ren_base import RENBase

batches = 4
nu, nx, nv, ny = 1, 1, 2, 1 # TODO: Allow nx = 0, and/or nv = 0
ren = RENBase(nx, nv, ny)

rng = jax.random.key(0)
key1, key2 = jax.random.split(rng)

states = ren.initialize_carry(key1, (batches, nu))
inputs = jnp.ones((batches, nu))

params = ren.init(key2, states, inputs)
print(jax.tree_map(jnp.shape, params))
