# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

from robustnn.plnet.monlipnet import MonLipNet
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
monlipnet_layer = MonLipNet(input_size=input_size, 
                            units=units,
                            mu=mu,
                            nu=nu,
                            tau=tau,
                            is_mu_fixed=False,
                            is_nu_fixed=False,
                            is_tau_fixed=True, 
                            )

# Initialize parameters
params = monlipnet_layer.init(key, inputs)
explicit_params = monlipnet_layer.direct_to_explicit(params)

# output
# parameters inside the model
# Parameters: {'params': {'logmu': Array([0.], dtype=float32), 'by': Array([0., 0.], dtype=float32), 'Fq': Array([[-0.67808104,  0.30136618,  0.15980107,  0.42550763],
#        [-0.44976526,  0.41513145,  0.10287036, -0.29770726]],      dtype=float32), 'fq': Array([1.1095239], dtype=float32), 'Fab0': Array([[-0.80490816, -0.13022006],
#        [ 0.8404852 ,  0.00755759]], dtype=float32), 'fab0': Array([1.1710281], dtype=float32), 'b0': Array([0., 0.], dtype=float32), 'Fab1': Array([[ 0.8647551 , -0.14875135],
#        [ 0.43040338,  0.32212225],
#        [-0.6120695 , -0.5580562 ],
#        [-0.09665938, -0.82898444]], dtype=float32), 'fab1': Array([1.5625466], dtype=float32), 'b1': Array([0., 0.], dtype=float32)}}
print("Parameters:", params)

# Bound:  (Array([1.], dtype=float32), Array([2.], dtype=float32), 2)
print("Bound: ", monlipnet_layer.get_bounds(params))

# Explicit Parameter: ExplicitMonLipParams(mu=Array([1.], dtype=float32), nu=Array([2.], dtype=float32), units=(2, 2), V=[Array([[ 0.8974826 ,  0.88322675],
#        [-1.5544597 ,  1.2352736 ]], dtype=float32)], S=Array([[ 0.6871649 , -0.2573519 ],
#        [ 0.15004435,  0.9252147 ],
#        [ 0.06253868, -0.3901289 ],
#        [ 0.4661716 , -0.8010758 ]], dtype=float32), by=Array([0., 0.], dtype=float32), bh=[], sqrt_g2=Array([0.70710677], dtype=float32), sqrt_2g=Array([1.4142135], dtype=float32), STks=[Array([[ 0.6871649 ,  0.15004435],
#        [-0.2573519 ,  0.9252147 ]], dtype=float32), Array([[ 0.06253868,  0.4661716 ],
#        [-0.3901289 , -0.8010758 ]], dtype=float32)], Ak_1s=[Array([], shape=(0, 0), dtype=float32), Array([[ 0.02972363, -0.99955815],
#        [ 0.99955815,  0.02972363]], dtype=float32)], BTks=[Array([], shape=(0, 2), dtype=float32), Array([[ 0.45475647,  0.5942618 ],
#        [-0.43541664,  0.7952448 ]], dtype=float32)], bs=[Array([0., 0.], dtype=float32), Array([0., 0.], dtype=float32)])
print("Explicit Parameter:", explicit_params)

# call results: [[1.9913986 2.4819925]
#  [5.3340416 4.9430366]]
print("call results:", monlipnet_layer.apply(params, random_input))

# explict call results: [[1.9913986 2.4819925]
#  [5.3340416 4.9430366]]
print("explict call results:", monlipnet_layer.explicit_call(params=params, x=random_input, explicit=explicit_params))
