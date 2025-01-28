"""NOTE:
Before running this test script, make sure that all initialisers
used by the networks are COMPLETELY deterministic.

Eg: change all instances of nn.init.xavier_normal_ to nn.init.ones_
in lbnn/lftn_pytorch.py.
"""

import jax, jax.numpy as jnp
from flax import linen
from robustnn.lftn_jax import LFTN as JX_LFTN

import torch
from robustnn.lftn_pytorch import FTN as PT_LFTN

nu = 5
ny = 2
nlayers = (8,16,8,ny)
gamma_jx = jnp.float32(2.0)
gamma_pt = 2.0

jx_model = JX_LFTN(
    layer_sizes=nlayers, 
    gamma=gamma_jx, 
    kernel_init=linen.initializers.ones_init())

params = jx_model.init(jax.random.key(0), jnp.ones((6,nu)))
u_jx = jnp.ones((4, nu))
y_jx = jx_model.apply(params, u_jx)

pt_model = PT_LFTN(nu, nlayers[:-1], ny, gamma=gamma_pt)
u_pt = torch.ones((4,nu))
y_pt = pt_model(u_pt)

print(y_jx)
print(y_pt)
