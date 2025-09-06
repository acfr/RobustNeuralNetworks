# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

from robustnn.plnet_torch.orthogonal import Unitary
import torch
import numpy as np

# Set seeds for all RNGs
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)        # for current GPU
torch.cuda.manual_seed_all(seed)    # for all GPUs
np.random.seed(seed)

# Generate random input
batches = 1
input_size = 2
inputs = torch.zeros((batches, input_size))
random_input = torch.tensor([[1.0,2.0],[3.0,4.0]])
units = [2,2]
depth = 2
mu = 1
nu = 2
tau = 2

# Initialize a unitary layer
unitary_layer = Unitary(in_features=input_size, out_features=input_size, bias=True)

# call results: tensor([[-1.7999,  1.6573]
        # [-3.1680,  4.1328]], grad_fn=<AddmmBackward0>)
print("call results:", unitary_layer( random_input))

# explict call results: [[-1.7998953  1.6573124]
#  [-3.168004   4.1328483]]
explict_call_res = unitary_layer.explicit_call( random_input.numpy(force=True), unitary_layer.direct_to_explicit())
print("explict call results:",  explict_call_res)

# inverse: [[1.        2.0000005]
#  [3.        4.       ]]
print("inverse:", unitary_layer.inverse(explict_call_res))