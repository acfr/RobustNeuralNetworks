from robustnn.plnet.bilipnet import BiLipNet
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
random_input = jax.numpy.array([[1,2],[3,4]])
units = [2,2]
mu = 1
nu = 2
tau = 2

# Initialize a unitary layer
bilipnet_layer = BiLipNet(input_size=input_size, 
                            units=units,
                            mu=mu,
                            nu=nu,
                            tau=tau,
                            is_mu_fixed=False,
                            is_nu_fixed=False,
                            is_tau_fixed=True, 
                            )

# Initialize parameters
params = bilipnet_layer.init(key, inputs)
explicit_params = bilipnet_layer.direct_to_explicit(params)

# output
# todo: errors here - result does not match the original!!!!!!!!!
# parameters inside the model
print("Parameters:", params)


print("Bound: ", bilipnet_layer.get_bounds(params))

print("Explicit Parameter:", explicit_params)

print("call results:", bilipnet_layer.apply(params, random_input))

print("explict call results:", bilipnet_layer.explicit_call(params=params, x=random_input, explicit=explicit_params))
