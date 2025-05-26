# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/nic-barbara/RobustNeuralNetworks/blob/main/LICENSE 

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

# Initialize parameters
params = bilipnet_layer.init(key, inputs)
explicit_params = bilipnet_layer.direct_to_explicit(params)

# output
# parameters inside the model
# Parameters: {'params': {'mon_0': {'by': Array([0., 0.], dtype=float32), 'Fq': Array([[-0.35264897,  0.4984822 , -0.47076553, -0.5240039 ],
#        [ 0.8253973 ,  0.18583922, -0.5114892 , -1.1876136 ]],      dtype=float32), 'fq': Array([1.8046912], dtype=float32), 'Fab0': Array([[-0.5248157 , -0.40291667],
#        [-0.78404427,  1.1737523 ]], dtype=float32), 'fab0': Array([1.5589076], dtype=float32), 'b0': Array([0., 0.], dtype=float32), 'Fab1': Array([[ 0.87369543, -0.2151519 ],
#        [ 0.49886596, -1.2565557 ],
#        [ 0.6950717 ,  0.331631  ],
#        [ 0.1621341 ,  0.02383308]], dtype=float32), 'fab1': Array([1.8048248], dtype=float32), 'b1': Array([0., 0.], dtype=float32)}, 'mon_1': {'by': Array([0., 0.], dtype=float32), 'Fq': Array([[ 0.6417317 ,  0.27034405, -1.1292694 , -0.35397807],
#        [ 0.30808014,  0.29465917, -0.1229675 , -0.5643722 ]],      dtype=float32), 'fq': Array([1.549461], dtype=float32), 'Fab0': Array([[-0.10914532,  0.28966933],
#        [-0.4470564 , -0.16007964]], dtype=float32), 'fab0': Array([0.56683856], dtype=float32), 'b0': Array([0., 0.], dtype=float32), 'Fab1': Array([[-0.5976882 , -0.52727705],
#        [-0.15430154,  0.30250213],
#        [ 0.5753947 ,  0.7232377 ],
#        [ 0.08675962,  0.22860019]], dtype=float32), 'fab1': Array([1.2901573], dtype=float32), 'b1': Array([0., 0.], dtype=float32)}, 'uni_0': {'W': Array([[-0.45691052,  0.27073306],
#        [ 0.30126202,  1.4242083 ]], dtype=float32), 'a': Array([1.5495778], dtype=float32), 'b': Array([0., 0.], dtype=float32)}, 'uni_1': {'W': Array([[ 1.3580841 , -0.20097394],
#        [ 0.79159266, -0.11509188]], dtype=float32), 'a': Array([1.588914], dtype=float32), 'b': Array([0., 0.], dtype=float32)}, 'uni_2': {'W': Array([[0.32905164, 1.3335835 ],
#        [0.71546155, 1.1022835 ]], dtype=float32), 'a': Array([1.900956], dtype=float32), 'b': Array([0., 0.], dtype=float32)}}}
print("Parameters:", params)

# Bound:  (1.0, 2.0000000000000004, 2.0000000000000004)
print("Bound: ", bilipnet_layer.get_bounds(params))

# Explicit Parameter: ExplicitBiLipParams(monlip_layers=[ExplicitMonLipParams(mu=1.0, nu=1.4142135623730951, units=(2, 2), V=[Array([[-0.8374117 ,  0.35082638],
#    [-1.3102895 ,  0.8824698 ]], dtype=float32)], S=Array([[ 0.22833245, -0.6014742 ],
#    [-0.6312865 ,  0.38985923],
#    [-0.17338762, -0.92611027],
#    [ 0.8809698 , -0.2251715 ]], dtype=float32), by=Array([0., 0.], dtype=float32), bh=[], sqrt_g2=Array(0.45508987, dtype=float32, weak_type=True), sqrt_2g=Array(0.91017973, dtype=float32, weak_type=True), STks=[Array([[ 0.22833245, -0.6312865 ],
#    [-0.6014742 ,  0.38985923]], dtype=float32), Array([[-0.17338762,  0.8809698 ],
#    [-0.92611027, -0.2251715 ]], dtype=float32)], Ak_1s=[Array([], shape=(0, 0), dtype=float32), Array([[ 0.74633104,  0.66557497],
#    [-0.66557497,  0.74633104]], dtype=float32)], BTks=[Array([], shape=(0, 2), dtype=float32), Array([[-0.42924377, -0.7826297 ],
#    [-0.14776382, -0.10674066]], dtype=float32)], bs=[Array([0., 0.], dtype=float32), Array([0., 0.], dtype=float32)]), ExplicitMonLipParams(mu=1.0, nu=1.4142135623730951, units=(2, 2), V=[Array([[-0.09556141,  0.6244243 ],
#    [-1.094813  ,  1.5568984 ]], dtype=float32)], S=Array([[-0.22651373,  0.45749414],
#    [ 0.07242526,  0.39718354],
#    [ 0.317936  , -0.7688174 ],
#    [-0.17970872,  0.18454075]], dtype=float32), by=Array([0., 0.], dtype=float32), bh=[], sqrt_g2=Array(0.45508987, dtype=float32, weak_type=True), sqrt_2g=Array(0.91017973, dtype=float32, weak_type=True), STks=[Array([[-0.22651373,  0.07242526],
#    [ 0.45749414,  0.39718354]], dtype=float32), Array([[ 0.317936  , -0.17970872],
#    [-0.7688174 ,  0.18454075]], dtype=float32)], Ak_1s=[Array([], shape=(0, 0), dtype=float32), Array([[ 0.29637387,  0.955072  ],
#    [-0.955072  ,  0.29637387]], dtype=float32)], BTks=[Array([], shape=(0, 2), dtype=float32), Array([[-0.312346  , -0.905712  ],
#    [ 0.04689751, -0.29210064]], dtype=float32)], bs=[Array([0., 0.], dtype=float32), Array([0., 0.], dtype=float32)])], unitary_layers=[ExplicitOrthogonalParams(R=Array([[ 0.9981377 ,  0.06100107],
#    [-0.06100107,  0.9981378 ]], dtype=float32), b=Array([0., 0.], dtype=float32)), ExplicitOrthogonalParams(R=Array([[ 0.00746098,  0.99997216],
#    [-0.9999722 ,  0.00746104]], dtype=float32), b=Array([0., 0.], dtype=float32)), ExplicitOrthogonalParams(R=Array([[ 0.44709978, -0.8944841 ],
#    [ 0.8944841 ,  0.44709978]], dtype=float32), b=Array([0., 0.], dtype=float32))], lipmin=1.0, lipmax=2.0000000000000004, distortion=2.0000000000000004)
print("Explicit Parameter:", explicit_params)

# call results: [[2.5733566 1.0719225]
#  [6.596445  1.4032273]]
print("call results:", bilipnet_layer.apply(params, random_input))

# explict call results: [[2.5733566 1.0719225]
#  [6.596445  1.4032273]]
print("explict call results:", bilipnet_layer.explicit_call(params=params, x=random_input, explicit=explicit_params))
