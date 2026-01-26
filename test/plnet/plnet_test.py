# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

from robustnn.bilipnet_jax import BiLipNet
from robustnn.plnet_jax import PLNet
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
depth = 2
mu = 1
nu = 2
tau = 2

# Initialize a unitary layer
bilipnet_layer = BiLipNet(input_size=input_size, 
                            units=units,
                            mu=mu,
                            nu=nu,
                            tau=tau,
                            is_mu_fixed=True,
                            is_nu_fixed=True,
                            is_tau_fixed=False, 
                            depth=depth,
                            act_fn=nn.relu,
                            )
plnet_layer = PLNet(BiLipBlock=bilipnet_layer)

# Initialize parameters
params = plnet_layer.init(key, inputs)
explicit_params = plnet_layer.direct_to_explicit(params)

# output
# Parameters: {'params': {'BiLipBlock': {'mon_0': {'by': Array([0., 0.], dtype=float32), 'Fq': Array([[ 0.4767888 ,  0.35946247,  0.88449204,  0.33577147],
#        [ 0.07560676,  0.04496887, -0.48031786,  0.04774935]],      dtype=float32), 'fq': Array([1.2216109], dtype=float32), 'Fab0': Array([[-0.72828573, -0.25070167],
#        [ 0.13120073, -1.6032633 ]], dtype=float32), 'fab0': Array([1.7835128], dtype=float32), 'b0': Array([0., 0.], dtype=float32), 'Fab1': Array([[ 0.9240948 ,  0.3495114 ],
#        [ 0.05895175,  1.1558645 ],
#        [-0.73868793, -0.30372804],
#        [ 0.37575445,  0.35212633]], dtype=float32), 'fab1': Array([1.7940742], dtype=float32), 'b1': Array([0., 0.], dtype=float32)}, 'mon_1': {'by': Array([0., 0.], dtype=float32), 'Fq': Array([[ 0.55890423,  0.71112454,  0.11298353, -0.07829883],
#        [-0.42325065, -0.45026565,  0.44604003, -0.03375688]],      dtype=float32), 'fq': Array([1.1911924], dtype=float32), 'Fab0': Array([[ 0.24955161,  0.1769813 ],
#        [ 0.41612598, -0.6867631 ]], dtype=float32), 'fab0': Array([0.85930365], dtype=float32), 'b0': Array([0., 0.], dtype=float32), 'Fab1': Array([[-0.05376392, -0.37428084],
#        [ 0.29803494, -0.3588037 ],
#        [ 0.9440928 ,  0.37645176],
#        [ 0.3438681 ,  0.56135315]], dtype=float32), 'fab1': Array([1.3516402], dtype=float32), 'b1': Array([0., 0.], dtype=float32)}, 'uni_0': {'W': Array([[ 0.8779757 ,  0.63369995],
#        [-0.06714347, -0.88191175]], dtype=float32), 'a': Array([1.3981035], dtype=float32), 'b': Array([0., 0.], dtype=float32)}, 'uni_1': {'W': Array([[-0.12902264,  0.7436014 ],
#        [ 0.17159157, -0.7215747 ]], dtype=float32), 'a': Array([1.0581604], dtype=float32), 'b': Array([0., 0.], dtype=float32)}, 'uni_2': {'W': Array([[ 0.94838375, -0.7258084 ],
#        [ 0.07415643, -0.43732023]], dtype=float32), 'a': Array([1.2739614], dtype=float32), 'b': Array([0., 0.], dtype=float32)}}}}
print("Parameters:", params)

# Explicit Parameter: ExplicitPLParams(bilip_layer=ExplicitBiLipParams(monlip_layers=[ExplicitMonLipParams(mu=1.0, nu=1.4142135623730951, units=(2, 2), V=[Array([[ 1.879422 ,  0.522046 ],
#        [ 0.5164222, -0.4752227]], dtype=float32)], S=Array([[-0.00249775,  0.00269252],
#        [ 0.14675875,  0.9206321 ],
#        [-0.2045758 , -0.16814956],
#        [ 0.37216145, -0.14749956]], dtype=float32), by=Array([0., 0.], dtype=float32), bh=[], sqrt_g2=Array(0.45508987, dtype=float32, weak_type=True), sqrt_2g=Array(0.91017973, dtype=float32, weak_type=True), STks=[Array([[-0.00249775,  0.14675875],
#        [ 0.00269252,  0.9206321 ]], dtype=float32), Array([[-0.2045758 ,  0.37216145],
#        [-0.16814956, -0.14749956]], dtype=float32)], Ak_1s=[Array([], shape=(0, 0), dtype=float32), Array([[ 0.74543  , -0.6665839],
#        [ 0.6665839,  0.74543  ]], dtype=float32)], BTks=[Array([], shape=(0, 2), dtype=float32), Array([[ 0.8744825 ,  0.03409042],
#        [-0.43182182, -0.349242  ]], dtype=float32)], bs=[Array([0., 0.], dtype=float32), Array([0., 0.], dtype=float32)]), ExplicitMonLipParams(mu=1.0, nu=1.4142135623730951, units=(2, 2), V=[Array([[-1.1694525 , -0.32475257],
#        [-0.32392204, -1.8697786 ]], dtype=float32)], S=Array([[ 0.39068747,  0.8540329 ],
#        [-0.8641052 ,  0.22347406],
#        [ 0.03718062,  0.4850111 ],
#        [-0.706138  ,  0.33923262]], dtype=float32), by=Array([0., 0.], dtype=float32), bh=[], sqrt_g2=Array(0.45508987, dtype=float32, weak_type=True), sqrt_2g=Array(0.91017973, dtype=float32, weak_type=True), STks=[Array([[ 0.39068747, -0.8641052 ],
#        [ 0.8540329 ,  0.22347406]], dtype=float32), Array([[ 0.03718062, -0.706138  ],
#        [ 0.4850111 ,  0.33923262]], dtype=float32)], Ak_1s=[Array([], shape=(0, 0), dtype=float32), Array([[ 0.8918072 , -0.45241565],
#        [ 0.45241565,  0.8918072 ]], dtype=float32)], BTks=[Array([], shape=(0, 2), dtype=float32), Array([[-0.5949247 , -0.5673965 ],
#        [ 0.11973099, -0.7604673 ]], dtype=float32)], bs=[Array([0., 0.], dtype=float32), Array([0., 0.], dtype=float32)])], unitary_layers=[ExplicitOrthogonalParams(R=Array([[ 0.34121832, -0.9399841 ],
#        [ 0.9399841 ,  0.34121832]], dtype=float32), b=Array([0., 0.], dtype=float32)), ExplicitOrthogonalParams(R=Array([[ 0.5069373 , -0.86198294],
#        [ 0.86198294,  0.5069373 ]], dtype=float32), b=Array([0., 0.], dtype=float32)), ExplicitOrthogonalParams(R=Array([[ 0.21955402,  0.97560036],
#        [-0.9756004 ,  0.21955411]], dtype=float32), b=Array([0., 0.], dtype=float32))], lipmin=1.0, lipmax=2.0000000000000004, distortion=2.0000000000000004), f_function=<function PLNet._direct_to_explicit.<locals>.f_function at 0x718d6b5ff6a0>, c=0.0, optimal_point=None, lipmin=1.0, lipmax=2.0000000000000004, distortion=2.0000000000000004)
print("Explicit Parameter:", explicit_params)

# call results: [ 2.73808  15.734728]
print("call results:", plnet_layer.apply(params, random_input))

# explict call results: [ 2.73808  15.734728]
print("explict call results:", plnet_layer.explicit_call(params=params, x=random_input, explicit=explicit_params))

# test for know equilibrium case
random_input_min = jax.numpy.array([[3,2],[4,1]])

# supposed to map to all dim
random_input_min_fixed = jax.numpy.array([[1,1]])

# explict call with gt:  [2.72025   3.4215217]
print("explict call with gt: ", plnet_layer.apply(params, random_input, random_input_min))

# explict call with mapping:  [0.3416915 7.868343 ]
print("explict call with mapping: ", plnet_layer.apply(params, random_input, random_input_min_fixed))
