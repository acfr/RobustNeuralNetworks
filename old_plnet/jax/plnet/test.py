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
from plnet.layer import BiLipNet
import jax.random as random
import orbax.checkpoint
from plnet.rosenbrock_utils import Sampler
from data_generators import two_dof_gen
import matplotlib.pyplot as plt
import scipy.io 
from plnet.train import data_gen, train
import os

import jax
import jax.numpy as jnp
import numpy as np

# %% [markdown]
# # Train 
# 

# %% [markdown]
# Default values

# %%
data_dim = 2
lr_max = 5e-3  # Reduced from 1e-2
epochs = 10 # 50 gives decent results
n_batch = 100  # Increased batch size for better gradient estimates
name = 'BiLipNet'
depth = 2
# layer_size = [128, 128, 128, 128]  # Reduced complexity
layer_size = [256]*8
tau = 40

# Use local path for results
root_dir = f'/home/RNN/rnn_ws/old_plnet/jax/plnet/results_exp/{name}-c-space-dim{data_dim}-batch{n_batch}'
os.makedirs(root_dir, exist_ok=True)

rng = random.PRNGKey(42)
rng, rng_data = random.split(rng, 2)

# %%
print("Generating dataset")
sampled_data = two_dof_gen.data_gen(rng_data, train_batches=n_batch, data_dim=data_dim, eval_batch_size=500, eval_batches=5)
print(f"Data_x: {sampled_data['xtrain'].shape}")
print(f"Data_y: {sampled_data['ytrain'].shape}")

# Normalize data to help with training
# Calculate mean and std for normalization
# x_mean = jnp.mean(sampled_data['xtrain'], axis=0)
# x_std = jnp.std(sampled_data['xtrain'], axis=0) + 1e-8
# y_mean = jnp.mean(sampled_data['ytrain'], axis=0)
# y_std = jnp.std(sampled_data['ytrain'], axis=0) + 1e-8

# # Normalize training data
# sampled_data['xtrain'] = (sampled_data['xtrain'] - x_mean) / x_std
# sampled_data['ytrain'] = (sampled_data['ytrain'] - y_mean) / y_std

# # Normalize test data
# sampled_data['xtest'] = (sampled_data['xtest'] - x_mean) / x_std
# sampled_data['ytest'] = (sampled_data['ytest'] - y_mean) / y_std

# # Normalize eval data
# sampled_data['xeval'] = (sampled_data['xeval'] - x_mean) / x_std
# sampled_data['yeval'] = (sampled_data['yeval'] - y_mean) / y_std

# # Store normalization parameters for later use
# sampled_data['x_mean'] = x_mean
# sampled_data['x_std'] = x_std
# sampled_data['y_mean'] = y_mean
# sampled_data['y_std'] = y_std

# %% [markdown]
# Train the model

# %%
# for tau in [tau]:
#     train_dir = f'{root_dir}/{name}-{depth}-tau{tau}'
#     model = BiLipNet(layer_size, depth=depth, tau=tau)
#     train(rng, model, sampled_data, name=name, train_dir=train_dir, lr_max=lr_max, epochs=epochs)

# %% [markdown]
# # Solve
# 

# %% [markdown]
# Restore the model

# %%
model = BiLipNet(layer_size, depth=depth, tau=tau)
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

train_dir = f'{root_dir}/{name}-{depth}-tau{tau}'
print(f"Testing dir: {train_dir}")
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
z = Sampler(rng, 100, 2)  # Reduced from 10000 to 100 for testing
# z = (z - x_mean) / x_std  # Apply the same normalization

# %%
from plnet.layer import Unitary
from plnet.layer import MonLipNet
import jax.numpy as jnp

(uni_params, mon_params, b_params, bh_params) = get_bilipnet_params(params,
                                                                    name='BiLipNet',
                                                                    depth=depth,
                                                                    orth=Unitary(),
                                                                    mln=MonLipNet(layer_size, jnp.sqrt(tau)))

# %%
import jax.numpy as jnp

# Apply the model to test data to see direct outputs
test_output = model.apply(params, sampled_data['xtest'])
print(f"Direct model outputs on test data:")
print(test_output)  # Print first 5 results
print(sampled_data['ytest'])  # Print first 5 results
'''
Example output
Direct model outputs on test data:
[[ 0.11635204  0.5589592 ]
 [ 0.523277    0.8457777 ]
 [-0.23916677  0.5241836 ]
 [-0.06529252 -0.05198373]
 [-0.20219782 -0.2716132 ]]
Truth:
[[-0.59180117  0.0159946 ]
 [ 0.31581658  0.8305541 ]
 [ 0.22115993  1.4178454 ]
 [ 0.7088315   1.6433233 ]
 [-1.9153016   0.2054239 ]]
'''

plt.figure(figsize=(10, 8))

# Create a normalized colormap based on index
num_points = len(sampled_data['ytest'])
colours = plt.cm.inferno(jnp.linspace(0, 1, num_points))

# import numpy as np

# Create a sorted index array based on increasing X then Y
sorted_indices = jnp.lexsort((sampled_data['ytest'][:, 0], sampled_data['ytest'][:, 1]))
# Apply the sorting to both arrays
sampled_data['ytest'] = sampled_data['ytest'][sorted_indices]
test_output = test_output[sorted_indices]

# Plot ground truth with 'o' markers
plt.scatter(sampled_data['ytest'][:, 0],
            sampled_data['ytest'][:, 1],
            c=colours, cmap=plt.cm.inferno, marker='o', label='Ground Truth')

# Plot model predictions with 'x' markers
plt.scatter(test_output[:, 0],
            test_output[:, 1],
            c=colours, cmap=plt.cm.inferno, marker='x', label='Model Predictions')

plt.legend()
plt.title('Configuration Space Mapping')
# plt.xlim(-2.1, 2.1)  # Set x-axis limits
# plt.ylim(-2.1, 2.1)  # Set y-axis limits
plt.savefig(f'{train_dir}/config_space_results.pdf')
plt.close()



z = Sampler(rng, 1000, 2)  # Reduced from 10000 to 100 for testing

# jitted_mln_back_solve_dys = jax.jit(mln_back_solve_dys)
data = mln_back_solve_dys(uni_params, mon_params, b_params, bh_params, z, depth,
                               layer_size, max_iter=max_iter, alpha=alpha, Lambda=Lambda,fn=fn)




print(f"DATA {data}")
# plt.semilogy(data['step'], data['vgap'])

# This code is ginving the error:
# Traceback (most recent call last):
#   File "/home/RNN/rnn_ws/old_plnet/jax/plnet/./test.py", line 196, in <module>
#     plt.semilogy(data['step'], data['vgap'])
#                  ~~~~^^^^^^^^
#   File "/home/RNN/.local/lib/python3.12/site-packages/jax/_src/array.py", line 382, in __getitem__
#     return indexing.rewriting_take(self, idx)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/RNN/.local/lib/python3.12/site-packages/jax/_src/numpy/indexing.py", line 632, in rewriting_take
#     treedef, static_idx, dynamic_idx = split_index_for_jit(idx, arr.shape)
#                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/RNN/.local/lib/python3.12/site-packages/jax/_src/numpy/indexing.py", line 720, in split_index_for_jit
#     raise TypeError(f"JAX does not support string indexing; got {idx=}")
# TypeError: JAX does not support string indexing; got idx=('step',)




# If data is a dictionary-like object with NumPy or regular Python arrays
# plt.semilogy(data['step'].to_numpy(), data['vgap'].to_numpy())  # for pandas
# # or
# plt.semilogy(np.array(data['step']), np.array(data['vgap']))  # for JAX arrays converted to NumPy

# # Or if you need to convert the whole JAX structure first
# # data_dict = {k: np.array(v) for k, v in data.items()}  # if data is a dict with JAX values
# # plt.semilogy(data_dict['step'], data_dict['vgap'])
# plt.savefig(f'{train_dir}/DYS-PLNet-alpha{alpha:.1f}-lambda{Lambda:.1f}.pdf')
# ```

# Solve the inverse problem
# data = mln_back_solve_dys(uni_params, mon_params, b_params, bh_params, , depth,
#                            layer_size, max_iter=max_iter, alpha=alpha, Lambda=Lambda, fn=fn)

# Plot ground truth with 'o' markers
plt.scatter(data[:, 0],
            data[:, 1],
            cmap=plt.cm.inferno, marker='o', label='Inverse model')

# Plot model predictions with 'x' markers
# plt.scatter(test_output[:, 0],
#             test_output[:, 1],
            # c=colours, cmap=plt.cm.inferno, marker='x', label='Model Predictions')

plt.legend()
plt.title('inverted Space Mapping')
# plt.xlim(-2.1, 2.1)  # Set x-axis limits
# plt.ylim(-2.1, 2.1)  # Set y-axis limits
plt.savefig(f'{train_dir}/inv_config_space_results.pdf')
plt.close()

# %% [markdown]
# Evaluate results

# Convert back from normalized space
# unnormalized_output = data * y_std + y_mean

# print(f"Test_x (first 5):\n{sampled_data['xtest'][:5] * x_std + x_mean}")
# print(f"Test_y (first 5):\n{sampled_data['ytest'][:5] * y_std + y_mean}")
# print(f"Model output y (first 5):\n{unnormalized_output[:5]}")


# Save results
# result_data = {
#     'test_x': sampled_data['xtest'] * x_std + x_mean,
#     'test_y': sampled_data['ytest'] * y_std + y_mean,
#     'model_output': test_output * y_std + y_mean,
#     'inverse_solution': unnormalized_output
# }
# scipy.io.savemat(f'{train_dir}/mapping_results.mat', result_data)
