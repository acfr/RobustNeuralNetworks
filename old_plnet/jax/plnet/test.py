#!/usr/bin/env python3
# %% [markdown]
# Write a simple example for dys with rosenbrock 
# 
# Created on 8/July-2024
# 

# %% [markdown]
# # dependency
#!/usr/bin/env python3

# %%
from plnet.solver import mln_back_solve_dys_demo, get_bilipnet_params, mln_back_solve_dys
from plnet.layer import BiLipNet, PLNet
import jax.random as random
import orbax.checkpoint
from plnet.rosenbrock_utils import Sampler
from data_generators import two_dof_gen
import matplotlib.pyplot as plt
import scipy.io 
from plnet.train import data_gen, train

import jax

# %% [markdown]
# # Train 
# 

# %% [markdown]
# Default values

# %%
data_dim = 2
lr_max = 1e-2
epochs = 100
n_batch = 50
name = 'BiLipNet'
depth = 2 
layer_size = [256]*8
tau=2

root_dir = f'results_exp/{name}-c-space-dim{data_dim}-batch{n_batch}'
rng = random.PRNGKey(42)
rng, rng_data = random.split(rng, 2)


# %%
print("Generating dataset")
data = two_dof_gen.data_gen(rng_data, train_batches=n_batch, data_dim=data_dim, eval_batch_size=500,eval_batches=5)
print(f"Data_x: {data['xtrain'].shape}")
print(f"Data_y: {data['ytrain'].shape}")
# print(data)
# print(data['xtrain'].shape)

# %% [markdown]
# train examples for rosenbrock with some configurations

# %%
# data= data_gen(rng_data, train_batches=n_batch, data_dim=data_dim)


for tau in [tau]:
    train_dir = f'{root_dir}/{name}-{depth}-tau{tau}'
    block = BiLipNet(layer_size, depth=depth, tau=tau)
    # Create model with 2D output for configuration space
    model = PLNet(block, output_dim=2)  # Match the dimension of ytrain (2D)
    train(rng, model, data, name=name, train_dir=train_dir, lr_max=lr_max, epochs=epochs)


# %% [markdown]
# # Solve
# 

# %% [markdown]
# Restore the model

# %%
model = PLNet(BiLipNet([256]*8, depth=depth, tau=tau), output_dim=2)
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

train_dir = f'{root_dir}/{name}-{depth}-tau{tau}'
print(f"testing dir: {train_dir}")
# where the param comes from
params = orbax_checkpointer.restore(f'{train_dir}/ckpt/params')

# run plnet func - maintain output dimensions
fn = lambda x, opt : model.apply(params, x)

# %% [markdown]
# solve the x based on given z

# %%
max_iter = 50
alpha = 1.0
Lambda = 1.0
rng = random.PRNGKey(43)
z = Sampler(rng, 10000, 20)

# %%
# print(params)
# print(params['params'])
# print(params['params']['BiLipBlock'])

# %%
from plnet.layer import Unitary
from plnet.layer import MonLipNet
import jax.numpy as jnp

(uni_params, mon_params, b_params, bh_params) = get_bilipnet_params(params, 
                                                                    depth = depth,
                                                                    orth=Unitary(),
                                                                    mln=MonLipNet(layer_size, jnp.sqrt(tau)))



# %%
import jax.numpy as jnp
# jitted_mln_back_solve_dys = jax.jit(mln_back_solve_dys)
# data = mln_back_solve_dys_demo(uni_params, mon_params, b_params, bh_params, z, fn, None,  
#                                layer_size, max_iter=max_iter, alpha=alpha, Lambda=Lambda, depth = depth, is_partial=False)
# data = jitted_mln_back_solve_dys(uni_params, mon_params, b_params, bh_params, z, depth,  
#                                layer_size, max_iter=max_iter, alpha=alpha, Lambda=Lambda,fn=fn)
data = mln_back_solve_dys(uni_params, mon_params, b_params, bh_params, jnp.zeros(jnp.shape(z)), depth,  
                               layer_size, max_iter=max_iter, alpha=alpha, Lambda=Lambda,fn=fn)

jax.debug.print('{}',data)
# %% [markdown]
# save

# # %%
# plt.semilogy(data['step'], data['vgap'])
# plt.savefig(f'{train_dir}/DYS-PLNet-alpha{alpha:.1f}-lambda{Lambda:.1f}.pdf')
# plt.close()
# scipy.io.savemat(f'{train_dir}/DYS-PLNet-alpha{alpha:.1f}-lambda{Lambda:.1f}.mat', data)

# %% [markdown]
# plot


