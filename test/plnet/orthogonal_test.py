# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

from robustnn.plnet.orthogonal import Unitary
from flax import linen as nn
import jax
import jax.numpy as jnp

# use some random key
rng = jax.random.key(0)
rng, key = jax.random.split(rng, 2)

# Generate random input
batches = 1
input_size = 2
inputs = jax.numpy.ones((batches, input_size))
random_input = jax.numpy.array([[1,2]])

# Initialize a unitary layer
unitary_layer = Unitary(input_size=input_size, use_bias=True)

# Initialize parameters
params = unitary_layer.init(key, inputs)
explicit_params = unitary_layer.direct_to_explicit(params)

# output
# parameters inside the model
# Parameters: {'params': {'W': Array([[-1.4166745 , -0.39387214],
#        [ 1.3889172 ,  0.8457249 ]], dtype=float32), 'a': Array([2.1923602], dtype=float32), 'b': Array([0., 0.], dtype=float32)}}
print("Parameters:", params)

# Explicit Parameter: ExplicitOrthogonalParams(R=Array([[-0.5213407,  0.8533486],
#        [-0.8533486, -0.5213408]], dtype=float32), b=Array([0., 0.], dtype=float32))
print("Explicit Parameter:", explicit_params)

# call results: [[ 1.1853565 -1.8960302]]
print("call results:", unitary_layer.apply(params, random_input))

# explict call results: [[ 1.1853565 -1.8960302]]
print("explict call results:", unitary_layer.explicit_call(params=params, x=random_input, explicit=explicit_params))
