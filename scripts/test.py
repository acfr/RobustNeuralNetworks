import jax
import jax.numpy as jnp

from robustnn.ren_jax.utils import tril_equlibrium_layer


##################### Test it all out #####################

D = jnp.array(
    [[0.0 , 0.        , 0.        , 0.        , 0.        ],
     [0.27276897, 0.0 , 0.        , 0.        , 0.        ],
     [0.8973534 , 0.45088673, 0.0, 0.        , 0.         ],
     [0.94310784, 0.02125645, 0.44761765, 0.0, 0.         ],
     [0.24344909, 0.17582   , 0.18456626, 0.40024185, 0.0 ]]
)
b = jnp.array(
    [[0.5338    , 0.9719182 , 0.61623883, 0.868845  , 0.6309322 ],
    [0.20438278, 0.7415488 , 0.15026295, 0.21696508, 0.32493377],
    [0.7355863 , 0.79253435, 0.3715024 , 0.1306243 , 0.04838264]]
)

# Run it once
act = jnp.tanh
w_eq = tril_equlibrium_layer(act, D, b)
print(w_eq)

# Test gradients
def loss(D, b):
    w_eq = tril_equlibrium_layer(act, D, b)
    return jnp.sum(w_eq**2)

grad_func = jax.jit(jax.grad(loss, argnums=(0,1)))
gs = grad_func(D,b)

print(loss(D,b))
print("Gradient for D: ", gs[0])
print("Gradient for b: ", gs[1])
