'''
The solver for dys
Author: Ruigang Wang
Edited: Dechuan

'''

import os 
import jax 
from jax import jacfwd, jacrev
import jax.numpy as jnp
from typing import Sequence, Callable
import scipy.io 
import matplotlib.pyplot as plt
from plnet.layer import *
import orbax.checkpoint
import numpy as np
import jax.random as random
from plnet.layer import PBiLipNet, PLNet, PUnitary, PMonLipNet, MonLipNet, Unitary, MLP
from typing import Any, Callable, Dict, Optional, Tuple, Union, Sequence
from functools import partial
#######################################################
## mln utils -------------------------------------------------------------
# -------------------------------------------------------------
# forward func


# param - R = cayley((a / jnp.linalg.norm(W)) * W)
# def orth_fwd(params, x):
#     return x @ params['R'].T + params['b'] 

# forward - eq 8
# def mln_fwd(params, x):
#     b = mln_fwd_x2b(params, x)
#     z = mln_fwd_b2z(params, b)
#     y = mln_fwd_xz2y(params, x, z)

#     return y

# z = vz+b
# def mln_fwd_b2z(params, b):
#     z = []
#     idx = 0
#     for k, nz in enumerate(params['units']):
#         if k == 0:
#             zk = nn.relu(b[..., idx:idx+nz])
#         else:
#             zk = nn.relu(zk @ params['V'][k-1].T + b[..., idx:idx+nz])
#         z.append(zk)
#         idx += nz 
#     return jnp.concatenate(z, axis=-1)

# mu x + sqrt(gamma / 2) S^T z + by
# def mln_fwd_xz2y(params, x, z):
#     return params['mu'] * x + jnp.sqrt(params['gam']/2) * z @ params['S'] + params['by']

# sqrt(2 gamma) x S^T + bh
# def mln_fwd_x2b(params, x):
#     return jnp.sqrt(2*params['gam']) * x @ params['S'].T + params['bh']

# -------------------------------------------------------------
# backward func
# (y-b)* R
def orth_bwd(R, y, b):
    return (y - b) @ R

# eq 8
# mu x + sqrt(gamma / 2) S^T z + by
def mln_bwd_yz2x(by, gam, S, mu, y, z):
    return (y - by - jnp.sqrt(gam/2) * z @ S) / mu

# eq 8
# sqrt(2 gamma) x S^T + bh
def mln_bwd_y2b(gam, mu, by, S, bh, y):
    return jnp.sqrt(2*gam)/mu * (y-by) @ S.T + bh

# C(z) in eq 14
# gamma / u * S * ST
def mln_bwd_z2v(gam, mu, S, z):
    return gam/mu * (z @ S) @ S.T 

# RA(v) in paper (eq 14 & 31) 
# ((1+a)I-aV)^(-1) (v+ab_z)
def mln_RA(gam, mu, S, V, alpha_, bz, zh, uh, units):
    # C(z)
    zv =  mln_bwd_z2v(gam, mu, S, zh)
    # eq 31
    # v=bz - gamma / u * S * ST
    vh = bz - zv

    au, av = 1/(1+alpha_), alpha_/(1+alpha_)
    # eq 31 a/(1+a)v + 1/(1+a)u
    b = av * vh + au * uh
    z = []
    idx = 0
    for k, nz in enumerate( units):
        if k == 0:
            zk = b[..., idx:idx+nz]
        else:
            # a/(1+a) V z + a/(1+a)v + 1/(1+a)u
            zk = av * zk @ V[k-1].T + b[..., idx:idx+nz]
        z.append(zk)
        idx += nz 
    return jnp.concatenate(z, axis=-1)



''' get the uni param and mon param for bi lip network
    input: params - params from entire model
    input: depth - depth of bi-lip
    input: unit - size of each monlip network
    input: tau - coefficients
    input: po_units - size of each mlp for b in unitary
    input: pb_units - size of each mlp for bh in monlip
    input: is_partial - do for partial bilip network
    return: a tuple of (uni_params, mon_params, b_params, bh_params)
'''
def get_bilipnet_params(params, depth, orth, mln,
               name: str = 'BiLipBlock') -> Tuple[Any, Any, Any, Any]:

    block_param = params['params']

    mon_params = {'mu':[], 'gam':[], 'units':[], 'V':[], 'S':[], 'bh':[], 'by':[]}
    uni_params = {'R':[], 'b':[]}
    b_params, bh_params= [], []
    for k in range(depth):
        # not seems to be setted anywhere else
        p = orth.apply({'params': block_param[f'uni_{k}']}, method=orth.get_params)
        uni_params['R'].append(p['R'])
        uni_params['b'].append(p['b'])
        b_params.append(p['b'])

        p = mln.apply({'params': block_param[f'mon_{k}']}, method=mln.get_params)
        mon_params['mu'].append(p['mu'])
        mon_params['gam'].append(p['gam'])
        mon_params['units'].append(p['units'])
        mon_params['V'].append(p['V'])
        mon_params['S'].append(p['S'])
        mon_params['bh'].append(p['bh'])
        mon_params['by'].append(p['by'])
        bh_params.append(p['bh'])


    p = orth.apply({'params': block_param[f'uni_{depth}']}, method=orth.get_params)
    uni_params['R'].append(p['R'])
    uni_params['b'].append(p['b'])
    b_params.append(p['b'])

    # After the loop, you can convert the lists in uni_params and mon_params to JAX arrays
    for key in uni_params:
        uni_params[key] = jnp.array(uni_params[key])

    for key in mon_params:
        mon_params[key] = jnp.array(mon_params[key])

    # Convert the separate lists to JAX arrays
    b_params = jnp.array(b_params)
    bh_params = jnp.array(bh_params)

    return (uni_params, mon_params, b_params, bh_params)


''' get the uni param and mon param for bi lip network
    input: params - params from entire model
    input: p_input - the fixed input p for the pplnet
    input: depth - depth of bi-lip
    input: unit - size of each monlip network
    input: tau - coefficients
    input: po_units - size of each mlp for b in unitary
    input: pb_units - size of each mlp for bh in monlip
    return: a tuple of (uni_params, mon_params, b_params, bh_params)
'''
def get_partial_bilipnet_params(params, p_input,
               tau: float = 10.0,
               depth: int = 2,
               unit: Sequence[int] = [256]*8,
               po_units: Sequence[int] = None,
               pb_units: Sequence[int] = None,
               name: str = 'PBiLipBlock') -> Tuple[Any, Any, Any, Any]:
    
    ltau = jnp.sqrt(tau)

    orth = PUnitary()
    mln = PMonLipNet(unit, tau=ltau)
    mlp_b = MLP(po_units)
    mlp_bh = MLP(pb_units)
    block_param = params['params'][name]

    mon_params = {'mu':[], 'gam':[], 'units':[], 'V':[], 'S':[], 'bh':[], 'by':[]}
    uni_params = {'R':[]}
    b_params, bh_params= [], []

    for k in range(depth):
        # not seems to be setted anywhere else
        p = orth.apply({'params': block_param[f'uni_{k}']}, method=orth.get_params)
        uni_params['R'].append(p['R'])
        b = mlp_b.apply({'params': block_param[f'uni_b_{k}']}, p_input)
        b_params.append(b)

        p = mln.apply({'params': block_param[f'mon_{k}']}, method=mln.get_params)
        mon_params['mu'].append(p['mu'])
        mon_params['gam'].append(p['gam'])
        mon_params['units'].append(p['units'])
        mon_params['V'].append(p['V'])
        mon_params['S'].append(p['S'])
        mon_params['by'].append(p['by'])
        bh = mlp_bh.apply({'params': block_param[f'mon_b_{k}']}, p_input)
        bh_params.append(bh)


    p = orth.apply({'params': block_param[f'uni_{depth}']}, method=orth.get_params)
    uni_params['R'].append(p['R'])
    b = mlp_b.apply({'params': block_param[f'uni_b_{depth}']}, p_input)
    b_params.append(b)

    for key in uni_params:
        uni_params[key] = jnp.array(uni_params[key])

    for key in mon_params:
        mon_params[key] = jnp.array(mon_params[key])

    # Convert the separate lists to JAX arrays
    b_params = jnp.array(b_params)
    bh_params = jnp.array(bh_params)

    return (uni_params, mon_params, b_params, bh_params)
##########################################################################################
# solver related
'''
Create dys func based on Lambda value
'''
def get_DYS_func(units: Sequence[int], Lambda: float = 1.0):
    # eq 14 in paper
    def DavisYinSplit(gam, mu, S, V, uk, bz, alpha_):
        # z = prox(u) = arg min 1/2|x-z|^2+af(z)
        zh = nn.relu(uk)
        # u=2z-u
        uh = 2*zh - uk 
        # eq 31
        # a/(1+a) V z + a/(1+a) (bz - gamma / u * S * ST zh) + 1/(1+a) uh
        zk = mln_RA(gam, mu, S, V, alpha_, bz, zh, uh, units)
        # u=u+z-z
        uk += Lambda * (zk - zh) 

        return zk, uk
    
    return DavisYinSplit

'''
condition function to check if reach maxi iteration
'''
def cond_fn(state):
    _, iter, iter_max, _, _ = state
    return iter < iter_max

def cond_fn_inverse(state):
    _, iter, iter_min = state
    return iter > iter_min

'''
check body func - running iteration  
# calculate the difference of x with a
input DavisYinSplit - the DYS function
input mon_params - monlip network params for current layer
input alphas - alpha value for current layer
return: a body function to run in iteration
'''
def get_body_fn( DavisYinSplit, gam, mu, S, V, alpha):
    def body_fn(state):
        _, iter, iter_max, uk, bz = state
        zk, uk = DavisYinSplit(gam, mu, S, V, uk, bz, alpha)
        return (zk, iter+1, iter_max, uk, bz)
    return body_fn  


''' 
solve by dys - cleaned for RL usage

can be called by
        (uni_params, mon_params, b_params, bh_params) = 
                        get_bilipnet_params(params, p, tau, depth, layer_size)
        data = mln_back_solve_dys_demo(uni_params, mon_params, b_params, bh_params, z, fn, 
                max_iter=max_iter, alpha=alpha, Lambda=Lambda, depth = depth)
input uni_params - unitary params with depth
input mon_params - monlip params with depth
...
return the best x input to make output to be 0
'''
def mln_back_solve_dys( uni_params, mon_params, b_params, bh_params, y_opt,
                        depth: int, 
                        units: Sequence[int],
                        max_iter: int = 500,
                        alpha: float = 1.0,
                        Lambda: float = 1.0,
                        fn = lambda z0, gt, : -1,
                        is_display = False):

    # get alphas for each mon layer
    indices = jnp.arange(depth)  # Create an array of indices for the loop
    # Define a function that computes one alpha for a given index
    def compute_alpha(i):
        return alpha * mon_params['mu'][i] / mon_params['gam'][i]
    # Vectorize the compute_alpha function across the indices
    alphas = jax.vmap(compute_alpha)(indices)

    # eq 14 in paper
    DavisYinSplit = get_DYS_func(units, Lambda)
    DavisYinSplit = jax.jit(DavisYinSplit)
    
    # k is the number of iterations for each layer
    def get_depth_loop_body_fn(k):
        # this is designed to go through all depth inverse related things
        # the only difference in each loop is the k value - number of iterations 
        # more iterations, get closer to 0 - in general 
        #################################################################
        # ? check with ray - might be possible to fix one and use it for all training (fast )
        
        def depth_loop_body_fn(state):
            x, iter, iter_max = state
            # 1 cycle for this depth level (0-3)
            y = orth_bwd(uni_params['R'][iter+1], x, b_params[iter+1])

            # monlip inverse
            bz = mln_bwd_y2b(mon_params['gam'][iter], 
                                mon_params['mu'][iter], 
                                mon_params['by'][iter],
                                mon_params['S'][iter],
                                bh_params[iter],
                                y)
            uk = jnp.zeros(jnp.shape(bz))
            
            # iterate until converge for z - find first z
            body_fn = get_body_fn( DavisYinSplit, mon_params['gam'][iter], 
                                    mon_params['mu'][iter], mon_params['S'][iter], 
                                    mon_params['V'][iter], alphas[iter])

            body_fn = jax.jit(body_fn)
            
            # (zk, iter+1, iter_max, uk, bz)
            zk = jax.lax.while_loop(cond_fn, body_fn, (uk, 0, k,  uk, bz))[0]

            # get x
            x = mln_bwd_yz2x(mon_params['by'][iter],
                                mon_params['gam'][iter], 
                                mon_params['S'][iter], 
                                mon_params['mu'][iter],
                                y, zk)
            # update
            iter -= 1
            return (x, iter, iter_max)
        return depth_loop_body_fn

    # set optimal z of output to be optimal y
    zopt = y_opt

    # modify the iterations could change the number of iteration executed by inverse solver
    iterations = jnp.arange(1, max_iter, 1)

    # doing inverse solver with k values from iterations
    def inverse_iteration(state):
        _, iter_i, max_iter, _, _ = state
        x = zopt
        k = iterations[iter_i]
        
        depth_loop_body_fn = get_depth_loop_body_fn(k)
        depth_loop_body_fn = jax.jit(depth_loop_body_fn)

        # go through each depth and get x to the first layer
        x = jax.lax.while_loop(cond_fn_inverse, depth_loop_body_fn, (x, depth-1, -1))[0]

        z0 = orth_bwd(uni_params['R'][0], x, b_params[0])

        if True:
            v = jnp.mean(fn(z0, y_opt))
            jax.debug.print('Iter. {} | v: {}', k, v)
            
        return ( z0, iter_i+1, max_iter, _, _)
    
    inverse_iteration = jax.jit(inverse_iteration)

    # loop to check each inverse
    z0 = jax.lax.while_loop(cond_fn, inverse_iteration, ( zopt, 0, iterations.size, 0, 0))[0]

    return z0      


''' 
solve by dys
build for test
can be called by
        (uni_params, mon_params, b_params, bh_params) = get_bilipnet_params(params, tau, depth, layer_size)
        data = mln_back_solve_dys_demo(uni_params, mon_params, b_params, bh_params, z, fn, 
                max_iter=max_iter, alpha=alpha, Lambda=Lambda, depth = depth)
input uni_params - unitary params with depth
input mon_params - monlip params with depth
return a data structure with the progress of learning {diff, stp, best_x}
'''
def mln_back_solve_dys_demo( uni_params, mon_params, b_params, bh_params,
                        z, fn, p, 
                        units: Sequence[int],
                        max_iter: int = 500,
                        alpha: float = 1.0,
                        Lambda: float = 1.0,
                        depth: int = 2, is_partial = True
):
    # get alphas for each mon layer
    alphas = []
    for i in range(depth):
        alphas.append(alpha*mon_params['mu'][i] / mon_params['gam'][i])

    # eq 14 in paper
    DavisYinSplit = get_DYS_func( units, Lambda)
    
    # pre-create body funcs - faster
    body_fns = [ get_body_fn( DavisYinSplit, 
                             mon_params['gam'][i], mon_params['mu'][i], 
                             mon_params['S'][i], mon_params['V'][i], 
                             alphas[i]) for i in range(depth)]
    
    vgap, step = [], []
    # assume optimal z is zero
    # z is some random number N~(0, 2)
    shp = jnp.shape(z)
    zopt = jnp.zeros(shp)
    k = 0
    while k <= max_iter:
        if k == 0:
            z0 = z
        else:
            x = zopt
            # traverse in reverse order
            for i in range( depth-1, -1, -1):
                # O inverse
                y = orth_bwd(uni_params['R'][i+1], x, b_params[i+1])

                # monlip inverse
                bz = mln_bwd_y2b(mon_params['gam'][i], 
                                 mon_params['mu'][i], 
                                 mon_params['by'][i],
                                 mon_params['S'][i],
                                 bh_params[i],
                                 y)
                uk = jnp.zeros(jnp.shape(bz))

                # iterate until converge for z - find first z
                zk = jax.lax.while_loop(cond_fn, body_fns[i], (uk, 0, k, uk, bz))[0]

                # get x
                x = mln_bwd_yz2x(mon_params['by'][i],
                                 mon_params['gam'][i], 
                                 mon_params['S'][i], 
                                 mon_params['mu'][i],
                                 y, zk)
                
            # last / first O layer (unitary)
            z0 = orth_bwd(uni_params['R'][0], x, b_params[0])

        # evaluate
        # evaluate
        if is_partial:
            v = jnp.mean(fn(z0, p))
        else:
            v = jnp.mean(fn(z0))
        vgap.append(v)
        step.append(k)
        print(f'Iter. {k:4d} | v: {v:.8f}')

        # next iteration
        if k <= 200:
            k += 1
        else:
            k += 10
    
    data = {
        'vgap': jnp.array(vgap),
        'step': jnp.array(step),
        'z': z0
    }

    return data         
