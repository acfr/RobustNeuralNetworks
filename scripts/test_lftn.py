import jax
from flax import linen
from jax import numpy as jnp
from lbnn.lftn_jax import LFTN

rng = jax.random.PRNGKey(0)
rng2 = jax.random.PRNGKey(1)

nu = 5
nlayers = (8,16,8,2)
model = LFTN(
    layer_sizes=nlayers,
    activation=linen.relu,
    gamma=jnp.float32(2.0)
)

params = model.init(rng, jnp.ones((6,nu)))
print(jax.tree_map(jnp.shape, params))

test = jax.random.normal(rng2, shape=(4, nu))
jax.debug.print("Out {x}", x = model.apply(params, test))