#!/usr/bin/env python3

'''
This file stores all training required function for rosenbrock

Created by Dechuan
Modified from Ruigang's code
'''
import sys
sys.path.append("..")

import jax.random as random
from plnet.rosenbrock_utils import Sampler, Rosenbrock, PRosenbrock
import os 
import jax 
import jax.numpy as jnp
import jax.random as random
import optax
import scipy.io 
from flax.training import train_state, orbax_utils
import orbax.checkpoint



def circular_data_generator(
    rng: random.PRNGKey,
    data_dim: int = 2, 
    val_min: float = -2.,
    val_max: float = 2.,
    train_batch_size: int = 200,
    test_batch_size: int = 5000,
    train_batches: int = 200,
    test_batches: int = 1,
    eval_batch_size: int = 5000,
    eval_batches: int = 100,
):
    # Calculate total points required
    total_train_points = train_batches * train_batch_size
    total_test_points = test_batches * test_batch_size
    total_eval_points = eval_batches * eval_batch_size
    total_points = total_train_points + total_test_points + total_eval_points

    # Generate random angles for circular dataset
    angles = random.uniform(rng, (total_points,), minval=0, maxval=2 * jnp.pi)
    
    # Generate coordinates based on circular pattern
    x1 = jnp.cos(angles)
    x2 = jnp.sin(angles)

    # Stack to create the input dataset
    X = jnp.stack([x1, x2], axis=1) * (val_max - val_min) + val_min

    # True dynamics: rotation vector field (circular motion)
    dx1 = -x2
    dx2 = x1
    dX = jnp.stack([dx1, dx2], axis=1) * (val_max - val_min)

    # Combine data into a single dataset for shuffling
    data = jnp.column_stack((X, dX))

    # Shuffle the dataset
    rng, shuffle_rng = random.split(rng)  # Split the RNG for shuffling
    shuffled_indices = random.permutation(shuffle_rng, jnp.arange(data.shape[0]))
    shuffled_data = data[shuffled_indices]

    # Split the shuffled dataset back into inputs and outputs
    X_shuffled = shuffled_data[:, :data_dim]  # Input features
    dX_shuffled = shuffled_data[:, data_dim:]  # True dynamics

    # Split the dataset into training, testing, and evaluation sets
    xtrain = X_shuffled[:total_train_points]
    ytrain = dX_shuffled[:total_train_points]
    
    xtest = X_shuffled[total_train_points:total_train_points + total_test_points]
    ytest = dX_shuffled[total_train_points:total_train_points + total_test_points]
    
    xeval = X_shuffled[total_train_points + total_test_points:]
    yeval = dX_shuffled[total_train_points + total_test_points:]

    data = {
        "xtrain": xtrain, 
        "ytrain": ytrain, 
        "xtest": xtest, 
        "ytest": ytest, 
        "xeval": xeval,
        "yeval": yeval,
        "train_batches": train_batches,
        "train_batch_size": train_batch_size,
        "test_batches": test_batches,
        "test_batch_size": test_batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "data_dim": data_dim
    }

    return data


def data_gen(
    rng: random.PRNGKey,
    data_dim: int = 20, 
    val_min: float = -2.,
    val_max: float = 2.,
    train_batch_size: int = 200,
    test_batch_size: int = 5000,
    train_batches: int = 200,
    test_batches: int = 1,
    eval_batch_size: int = 5000,
    eval_batches: int = 100,
):
    rng_train, rng_test, rng_eval = random.split(rng, 3)
    
    xtrain = Sampler(rng_train, 
                     train_batch_size * train_batches,
                     data_dim, 
                     x_min=val_min, x_max=val_max)
    xtest  = Sampler(rng_test, 
                     test_batch_size * test_batches,
                     data_dim, 
                     x_min=val_min, x_max=val_max)
    xeval  = Sampler(rng_eval, 
                     eval_batch_size * eval_batches, 
                     data_dim, 
                     x_min=val_min, x_max=val_max)

    ytrain, ytest, yeval = Rosenbrock(xtrain), Rosenbrock(xtest), Rosenbrock(xeval)
    
    data = {
        "xtrain": xtrain, 
        "ytrain": ytrain, 
        "xtest": xtest, 
        "ytest": ytest, 
        "xeval": xeval,
        "yeval": yeval,
        "train_batches": train_batches,
        "train_batch_size": train_batch_size,
        "test_batches": test_batches,
        "test_batch_size": test_batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "data_dim": data_dim
    }

    return data

'''
generate data for partial network
'''
def data_gen_partial(
    rng: random.PRNGKey,
    data_dim: int = 20, 
    val_min: float = -2.,
    val_max: float = 2.,
    train_batch_size: int = 200,
    test_batch_size: int = 5000,
    train_batches: int = 200,
    test_batches: int = 1,
    eval_batch_size: int = 5000,
    eval_batches: int = 100,
):
    rng_train, rng_test, rng_eval = random.split(rng, 3)
    
    xtrain = Sampler(rng_train, 
                     train_batch_size * train_batches,
                     data_dim, 
                     x_min=val_min, x_max=val_max)
    xtest  = Sampler(rng_test, 
                     test_batch_size * test_batches,
                     data_dim, 
                     x_min=val_min, x_max=val_max)
    xeval  = Sampler(rng_eval, 
                     eval_batch_size * eval_batches, 
                     data_dim, 
                     x_min=val_min, x_max=val_max)
    
	# p [a,b]
    ptrain = Sampler(rng_train, 
                     train_batch_size * train_batches,
                     2, 
                     x_min=val_min, x_max=val_max)
    ptest  = Sampler(rng_test, 
                     test_batch_size * test_batches,
                     2, 
                     x_min=val_min, x_max=val_max)
    peval  = Sampler(rng_eval, 
                     eval_batch_size * eval_batches, 
                     2, 
                     x_min=val_min, x_max=val_max)

    ytrain, ytest, yeval = PRosenbrock(xtrain, ptrain), PRosenbrock(xtest, ptest), PRosenbrock(xeval, peval)
    
    data = {
        "xtrain": xtrain, 
        "ptrain": ptrain, 
        "ytrain": ytrain, 
        "xtest": xtest, 
        "ptest": ptest, 
        "ytest": ytest, 
        "xeval": xeval,
        "peval": peval,
        "yeval": yeval,
        "train_batches": train_batches,
        "train_batch_size": train_batch_size,
        "test_batches": test_batches,
        "test_batch_size": test_batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "data_dim": data_dim,
        "p_dim": 2
    }

    return data


'''
train the model for partial input
'''
def train_partial(
    rng,
    model,
    data,
    name: str = 'pbilipnet',
    train_dir: str = './results/rosenbrock-nd',
    lr_max: float = 1e-3,
    epochs: int = 600,
):

    ckpt_dir = f'{train_dir}/ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dim = data['data_dim']
    p_dim = data['p_dim']
    train_batches = data['train_batches']
    train_batch_size = data['train_batch_size']

    idx_shp = (train_batches, train_batch_size)
    train_size = train_batches * train_batch_size

    rng, rng_model = random.split(rng)

    params = model.init(rng_model, jnp.ones(data_dim), jnp.ones(p_dim))

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'model: {name}, size: {param_count/1000000:.2f}M')

    total_steps = train_batches * epochs
    scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
                                           peak_value=lr_max,
                                           pct_start=0.25, 
                                           pct_final=0.7,
                                           div_factor=10., 
                                           final_div_factor=200.)
    opt = optax.adam(learning_rate=scheduler)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=opt)
    
    @jax.jit
    def fitloss(state, params, x, p, y):
        yh = state.apply_fn(params, x, p)
        loss = optax.l2_loss(yh, y).mean()
        return loss
    
    @jax.jit
    def train_step(state, x, p, y):
        grad_fn = jax.value_and_grad(fitloss, argnums=1)
        loss, grads = grad_fn(state, state.params, x, p, y)
        state = state.apply_gradients(grads=grads)
        return state, loss 
    
    train_loss, val_loss = [], []
    Lipmin, Lipmax, Tau = [], [], []
    for epoch in range(epochs):
        rng, rng_idx = random.split(rng)
        idx = random.permutation(rng_idx, train_size)
        idx = jnp.reshape(idx, idx_shp)
        tloss = 0. 
        for b in range(train_batches):
            x = data['xtrain'][idx[b, :], :] 
            y = data['ytrain'][idx[b, :]]
            p = data['ptrain'][idx[b, :]]
            model_state, loss = train_step(model_state, x, p, y)
            tloss += loss
        tloss /= train_batches
        train_loss.append(tloss)

        vloss = fitloss(model_state, model_state.params, data['xtest'], data['ptest'], data['ytest'])
        val_loss.append(vloss)

        lipmin, lipmax, tau = model.apply(model_state.params, method=model.get_bounds)
        Lipmin.append(lipmin)
        Lipmax.append(lipmax)
        Tau.append(tau)

        print(f'Epoch: {epoch+1:3d} | loss: {tloss:.4f}/{vloss:.4f}, tau: {tau:.1f}, Lip: {lipmin:.3f}/{lipmax:.2f}')

    eloss = fitloss(model_state, model_state.params, data['xeval'], data['peval'], data['yeval'])
    print(f'{name}: eval loss: {eloss:.4f}')

    data['train_loss'] = jnp.array(train_loss)
    data['val_loss'] = jnp.array(val_loss)
    data['lipmin'] = jnp.array(Lipmin)
    data['lipmax'] = jnp.array(Lipmax)
    data['tau'] = jnp.array(Tau)
    data['eval_loss'] = eloss

    scipy.io.savemat(f'{train_dir}/data.mat', data)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model_state.params)
    orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)

def get_fitness_loss(model, 
                     is_loss_tau: float = False,
                     is_loss_mu: float = False,
                     gam_logtau: float = 0.1,
                     gam_logmu: float = 0.1, # minus this value and penalize it when too small
                     ):
    if is_loss_mu == False and is_loss_tau == False:
        @jax.jit
        def fitloss(state, params, x, y):
            yh = state.apply_fn(params, x)
            # print(f"state: {state}")
            # print(f"params: {params}")
            # print(f"x shape: {x.shape}")
            # print(f"yh shape: {yh.shape}")
            # print(f"y shape: {y.shape}")
            loss = optax.l2_loss(yh, y).mean()
            return loss
    elif is_loss_mu == False and is_loss_tau == True:
        @jax.jit
        def fitloss(state, params, x, y):
            yh = state.apply_fn(params, x)
            logtau = model.apply(params, method=model.get_logtau)
            loss_v = optax.l2_loss(yh, y).mean()
            loss = loss_v + gam_logtau*logtau
            return loss
    elif is_loss_mu == True and is_loss_tau == True:
        @jax.jit
        def fitloss(state, params, x, y):
            yh = state.apply_fn(params, x)
            logtau = model.apply(params, method=model.get_logtau)
            logmu = model.apply(params, method=model.get_logmu)
            loss_v = optax.l2_loss(yh, y).mean()
            loss = loss_v + gam_logtau*logtau - gam_logmu * logmu
            return loss
    else:
        @jax.jit
        def fitloss(state, params, x, y):
            yh = state.apply_fn(params, x)
            logmu = model.apply(params, method=model.get_logmu)
            loss_v = optax.l2_loss(yh, y).mean()
            loss = loss_v - gam_logmu * logmu
            return loss
    return fitloss

'''
train the model
'''
def train(
    rng,
    model,
    data,
    name: str = 'bilipnet',
    train_dir: str = './results/rosenbrock-nd',
    lr_max: float = 1e-3,
    epochs: int = 600,
    is_loss_tau: float = False,
    is_loss_mu: float = False,
    gam_logtau: float = 1e-4,
    gam_logmu: float = 1e-4, # minus this value and penalize it when too small
    update_id: int = 0
):

    ckpt_dir = f'{train_dir}/ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dim = data['data_dim']
    train_batches = data['train_batches']
    train_batch_size = data['train_batch_size']

    idx_shp = (train_batches, train_batch_size)
    train_size = train_batches * train_batch_size

    rng, rng_model = random.split(rng)
    params = model.init(rng_model, jnp.ones(data_dim))
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'model: {name}, size: {param_count/1000000:.2f}M')

    total_steps = train_batches * epochs
    scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
                                           peak_value=lr_max,
                                           pct_start=0.25, 
                                           pct_final=0.7,
                                           div_factor=10., 
                                           final_div_factor=200.)
    opt = optax.adam(learning_rate=scheduler)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=opt)
    
    fitloss = get_fitness_loss(model=model, 
                               is_loss_tau=is_loss_tau,
                               is_loss_mu=is_loss_mu,
                               gam_logtau=gam_logtau,
                               gam_logmu=gam_logmu)
    
    fitloss_loss = get_fitness_loss(model=model, 
                               is_loss_tau=False,
                               is_loss_mu=False,
                               gam_logtau=gam_logtau,
                               gam_logmu=gam_logmu)
    
    @jax.jit
    def train_step(state, x, y):
        grad_fn = jax.value_and_grad(fitloss, argnums=1)
        loss, grads = grad_fn(state, state.params, x, y)
        state = state.apply_gradients(grads=grads)
        return state, loss 
    
    train_loss, val_loss = [], []
    Lipmin, Lipmax, Tau = [], [], []
    for epoch in range(epochs):
        rng, rng_idx = random.split(rng)
        idx = random.permutation(rng_idx, train_size)
        idx = jnp.reshape(idx, idx_shp)
        tloss = 0. 
        for b in range(train_batches):
            x = data['xtrain'][idx[b, :], :]
            # print(f"TRAINING_x: {x}")
            y = data['ytrain'][idx[b, :]]
            # print(f"TRAINING_y: {y}")
            model_state, loss = train_step(model_state, x, y)
            tloss += loss
        tloss /= train_batches
        train_loss.append(tloss)

        vloss = fitloss_loss(model_state, model_state.params, data['xtest'], data['ytest'])
        val_loss.append(vloss)

        lipmin, lipmax, tau = model.apply(model_state.params, method=model.get_bounds)
        Lipmin.append(lipmin)
        Lipmax.append(lipmax)
        Tau.append(tau)

        print(f'Epoch: {epoch+1:3d} | loss: {tloss:.4f}/{vloss:.4f}, tau: {tau:.1f}, Lip: {lipmin:.3f}/{lipmax:.2f}')

    eloss = fitloss_loss(model_state, model_state.params, data['xeval'], data['yeval'])
    print(f'{name}: eval loss: {eloss:.4f}')

    data['train_loss'] = jnp.array(train_loss)
    data['val_loss'] = jnp.array(val_loss)
    data['lipmin'] = jnp.array(Lipmin)
    data['lipmax'] = jnp.array(Lipmax)
    data['tau'] = jnp.array(Tau)
    data['eval_loss'] = eloss

    scipy.io.savemat(f'{train_dir}/data.mat', data)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model_state.params)
    orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)

'''
train the model
'''
def train_with_flexible_loss(
    rng,
    model,
    data,
    fitness_func,
    fitness_eval_func,
    name: str = 'bilipnet',
    train_dir: str = './results/rosenbrock-nd',
    lr_max: float = 1e-3,
    epochs: int = 600,
    update_id: int = 0,
    is_batched_input: bool= False,
):

    ckpt_dir = f'{train_dir}/ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dim = data['data_dim']
    train_batches = data['train_batches']
    train_batch_size = data['train_batch_size']

    idx_shp = (train_batches, train_batch_size)
    train_size = train_batches * train_batch_size

    rng, rng_model = random.split(rng)
    
    # special batch considered for J and R
    if is_batched_input:
        params = model.init(rng_model, jnp.ones((1, data_dim)))
    else:
        params = model.init(rng_model, jnp.ones(data_dim))
    
    # jax.debug.print("init params: {}", params)
        
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'model: {name}, size: {param_count/1000000:.2f}M')

    total_steps = train_batches * epochs
    scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
                                           peak_value=lr_max,
                                           pct_start=0.25, 
                                           pct_final=0.7,
                                           div_factor=10., 
                                           final_div_factor=200.)
    opt = optax.adam(learning_rate=scheduler)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=opt)
    
    @jax.jit
    def train_step(state, x, y):
        grad_fn = jax.value_and_grad(fitness_func, argnums=1)
        loss, grads = grad_fn(state, state.params, x, y)
        # jax.debug.print("loss: {}\ngrads: {}", loss, grads)
        state = state.apply_gradients(grads=grads)
        return state, loss 
    
    train_loss, val_loss = [], []
    Lipmin, Lipmax, Tau = [], [], []
    for epoch in range(epochs):
        rng, rng_idx = random.split(rng)
        idx = random.permutation(rng_idx, train_size)
        idx = jnp.reshape(idx, idx_shp)
        tloss = 0. 
        for b in range(train_batches):
            x = data['xtrain'][idx[b, :], :] 
            y = data['ytrain'][idx[b, :]]
            model_state, loss = train_step(model_state, x, y)
            tloss += loss
        tloss /= train_batches
        train_loss.append(tloss)

        vloss = fitness_eval_func(model_state, model_state.params, data['xtest'], data['ytest'])
        val_loss.append(vloss)

        lipmin, lipmax, tau = model.apply(model_state.params, method=model.get_bounds)
        Lipmin.append(lipmin)
        Lipmax.append(lipmax)
        Tau.append(tau)

        print(f'Epoch: {epoch+1:3d} | loss: {tloss:.4f}/{vloss[-1]}, tau: {tau:.1f}, Lip: {lipmin:.3f}/{lipmax:.2f}')

    eloss = fitness_eval_func(model_state, model_state.params, data['xeval'], data['yeval'])
    print(f'{name}: eval loss: {eloss[-1]}')

    data['train_loss'] = jnp.array(train_loss)
    data['val_loss'] = jnp.array(val_loss)
    data['lipmin'] = jnp.array(Lipmin)
    data['lipmax'] = jnp.array(Lipmax)
    data['tau'] = jnp.array(Tau)
    data['eval_loss'] = eloss

    scipy.io.savemat(f'{train_dir}/data.mat', data)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model_state.params)
    orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)


'''
train the model with optimal
'''
def train_with_optimal(
    rng,
    model,
    data,
    name: str = 'bilipnet',
    train_dir: str = './results/rosenbrock-nd',
    lr_max: float = 1e-3,
    epochs: int = 600,
    optimal_func = None
):
    if optimal_func == None:
        optimal_func = lambda obs: jnp.zeros(jnp.shape(obs))

    ckpt_dir = f'{train_dir}/ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dim = data['data_dim']
    train_batches = data['train_batches']
    train_batch_size = data['train_batch_size']

    idx_shp = (train_batches, train_batch_size)
    train_size = train_batches * train_batch_size

    rng, rng_model = random.split(rng)
    params = model.init(rng_model, jnp.ones(data_dim), jnp.ones(data_dim))
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'model: {name}, size: {param_count/1000000:.2f}M')

    total_steps = train_batches * epochs
    scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
                                           peak_value=lr_max,
                                           pct_start=0.25, 
                                           pct_final=0.7,
                                           div_factor=10., 
                                           final_div_factor=200.)
    opt = optax.adam(learning_rate=scheduler)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=opt)
    
    @jax.jit
    def fitloss(state, params, x, x_opt, y):
        yh = state.apply_fn(params, x, x_opt)
        loss = optax.l2_loss(yh, y).mean()
        return loss
    
    @jax.jit
    def train_step(state, x, y):
        grad_fn = jax.value_and_grad(fitloss, argnums=1)
        loss, grads = grad_fn(state, state.params, x, optimal_func(x), y)
        state = state.apply_gradients(grads=grads)
        return state, loss 
    
    train_loss, val_loss = [], []
    Lipmin, Lipmax, Tau = [], [], []
    for epoch in range(epochs):
        rng, rng_idx = random.split(rng)
        idx = random.permutation(rng_idx, train_size)
        idx = jnp.reshape(idx, idx_shp)
        tloss = 0. 
        for b in range(train_batches):
            x = data['xtrain'][idx[b, :], :] 
            y = data['ytrain'][idx[b, :]]
            model_state, loss = train_step(model_state, x, y)
            tloss += loss
        tloss /= train_batches
        train_loss.append(tloss)

        vloss = fitloss(model_state, model_state.params, data['xtest'], optimal_func(data['xtest']), data['ytest'])
        val_loss.append(vloss)

        lipmin, lipmax, tau = model.apply(model_state.params, method=model.get_bounds)
        Lipmin.append(lipmin)
        Lipmax.append(lipmax)
        Tau.append(tau)

        print(f'Epoch: {epoch+1:3d} | loss: {tloss:.4f}/{vloss:.4f}, tau: {tau:.1f}, Lip: {lipmin:.3f}/{lipmax:.2f}')

    eloss = fitloss(model_state, model_state.params, data['xeval'], optimal_func(data['xeval']), data['yeval'])
    print(f'{name}: eval loss: {eloss:.4f}')

    data['train_loss'] = jnp.array(train_loss)
    data['val_loss'] = jnp.array(val_loss)
    data['lipmin'] = jnp.array(Lipmin)
    data['lipmax'] = jnp.array(Lipmax)
    data['tau'] = jnp.array(Tau)
    data['eval_loss'] = eloss

    scipy.io.savemat(f'{train_dir}/data.mat', data)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model_state.params)
    orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)


'''
train the model with optimal
take the loss function as input to allow more flexibility
eg for optimal func:
        optimal_func = lambda obs: jnp.zeros(jnp.shape(obs))
eg for fitness_func:
        @jax.jit
        def fitloss(state, params, x, y):
            yh = state.apply_fn(params, x)
            logtau = model.apply(state.params, method=model.get_logtau)
            logmu = model.apply(state.params, method=model.get_logmu)
            loss_v = optax.l2_loss(yh, y).mean()
            loss = loss_v + gam_logtau*logtau - gam_logmu * logmu
            return loss
'''
def train_with_optimal_flexible_loss(
    rng,
    model,
    data,
    fitness_func,
    fitness_eval_func,
    name: str = 'bilipnet',
    train_dir: str = './results/rosenbrock-nd',
    lr_max: float = 1e-3,
    epochs: int = 600,
    weight_update_func = None, # define if the weight is added for the fitness_func(state, param, x, y, weight) & weight update
):
    # define weight matrix
    is_adding_weight = True
    if weight_update_func == None:
        is_adding_weight = False

    ckpt_dir = f'{train_dir}/ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dim = data['data_dim']
    train_batches = data['train_batches']
    train_batch_size = data['train_batch_size']

    idx_shp = (train_batches, train_batch_size)
    train_size = train_batches * train_batch_size

    rng, rng_model = random.split(rng)
    params = model.init(rng_model, jnp.ones(data_dim), jnp.ones(data_dim))
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'model: {name}, size: {param_count/1000000:.2f}M')

    total_steps = train_batches * epochs
    scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
                                           peak_value=lr_max,
                                           pct_start=0.25, 
                                           pct_final=0.7,
                                           div_factor=10., 
                                           final_div_factor=200.)
    opt = optax.adam(learning_rate=scheduler)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=opt)
    if is_adding_weight:
        weight_matrix = jnp.array(jnp.ones_like(data['ytrain']))

    @jax.jit
    def train_step(state, x, y):
        grad_fn = jax.value_and_grad(fitness_func, argnums=1)
        loss, grads = grad_fn(state, state.params, x, y)
        state = state.apply_gradients(grads=grads)
        return state, loss 
    
    @jax.jit
    def train_step_weighted(state, x, y, weight_matrix_batch):
        grad_fn = jax.value_and_grad(fitness_func, argnums=1)
        loss, grads = grad_fn(state, state.params, x, y, weight_matrix_batch)
        state = state.apply_gradients(grads=grads)
        return state, loss 
    
    train_loss, val_loss = [], []
    Lipmin, Lipmax, Tau = [], [], []
    for epoch in range(epochs):
        rng, rng_idx = random.split(rng)
        idx = random.permutation(rng_idx, train_size)
        idx = jnp.reshape(idx, idx_shp)
        tloss = 0. 
        for b in range(train_batches):
            x = data['xtrain'][idx[b, :], :] 
            y = data['ytrain'][idx[b, :]]

            # give weights back into fitness func
            if is_adding_weight:
                weight_matrix_batch = weight_matrix[idx[b, :]]
                model_state, loss = train_step_weighted(model_state, x, y, weight_matrix_batch)
            else:
                model_state, loss = train_step(model_state, x, y)

            tloss += loss
        tloss /= train_batches
        train_loss.append(tloss)

        # update weight matrix
        if is_adding_weight:
            weight_matrix = weight_update_func(model_state.params, data['xtrain'], data['ytrain'], weight_matrix)

        # validation
        vloss = fitness_eval_func(model_state, model_state.params, data['xtest'], data['ytest'])
        val_loss.append(vloss)

        lipmin, lipmax, tau = model.apply(model_state.params, method=model.get_bounds)
        Lipmin.append(lipmin)
        Lipmax.append(lipmax)
        Tau.append(tau)

        print(f'Epoch: {epoch+1:3d} | loss: {tloss:.4f}/{vloss[-1]:.4f}, tau: {tau:.1f}, Lip: {lipmin:.3f}/{lipmax:.2f}')

    eloss = fitness_eval_func(model_state, model_state.params, data['xeval'], data['yeval'])
    print(f'{name}: eval loss: {eloss[-1]}')

    data['train_loss'] = jnp.array(train_loss)
    data['val_loss'] = jnp.array(val_loss)
    data['lipmin'] = jnp.array(Lipmin)
    data['lipmax'] = jnp.array(Lipmax)
    data['tau'] = jnp.array(Tau)
    data['eval_loss'] = eloss

    scipy.io.savemat(f'{train_dir}/data.mat', data)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model_state.params)
    orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)


# def train(
#     rng,
#     model,
#     data,
#     name: str = 'bilipnet',
#     train_dir: str = './results/rosenbrock-nd',
#     lr_max: float = 1e-3,
#     epochs: int = 600,
# ):

#     ckpt_dir = f'{train_dir}/ckpt'
#     os.makedirs(ckpt_dir, exist_ok=True)

#     data_dim = data['data_dim']
#     train_batches = data['train_batches']
#     train_batch_size = data['train_batch_size']

#     idx_shp = (train_batches, train_batch_size)
#     train_size = train_batches * train_batch_size

#     rng, rng_model = random.split(rng)
#     params = model.init(rng_model, jnp.ones(data_dim), jnp.ones(2))
#     param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
#     print(f'model: {name}, size: {param_count/1000000:.2f}M')

#     total_steps = train_batches * epochs
#     scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
#                                            peak_value=lr_max,
#                                            pct_start=0.25, 
#                                            pct_final=0.7,
#                                            div_factor=10., 
#                                            final_div_factor=200.)
#     opt = optax.adam(learning_rate=scheduler)
#     model_state = train_state.TrainState.create(apply_fn=model.apply,
#                                                 params=params,
#                                                 tx=opt)
    
#     @jax.jit
#     def fitloss(state, params, x, p, y):
#         yh = state.apply_fn(params, x, p)
#         loss = optax.l2_loss(yh, y).mean()
#         return loss
    
#     @jax.jit
#     def train_step(state, x, p, y):
#         grad_fn = jax.value_and_grad(fitloss, argnums=1)
#         loss, grads = grad_fn(state, state.params, x, p, y)
#         state = state.apply_gradients(grads=grads)
#         return state, loss 
    
#     train_loss, val_loss = [], []
#     Lipmin, Lipmax, Tau = [], [], []
#     for epoch in range(epochs):
#         rng, rng_idx = random.split(rng)
#         idx = random.permutation(rng_idx, train_size)
#         idx = jnp.reshape(idx, idx_shp)
#         tloss = 0. 
#         for b in range(train_batches):
#             x = data['xtrain'][idx[b, :], :] 
#             p = data['ptrain'][idx[b, :], :] 
#             y = data['ytrain'][idx[b, :]]
#             model_state, loss = train_step(model_state, x, p, y)
#             tloss += loss
#         tloss /= train_batches
#         train_loss.append(tloss)

#         vloss = fitloss(model_state, model_state.params, data['xtest'], data['ptest'], data['ytest'])
#         val_loss.append(vloss)

#         lipmin, lipmax, tau = model.apply(model_state.params, method=model.get_bounds)
#         Lipmin.append(lipmin)
#         Lipmax.append(lipmax)
#         Tau.append(tau)

#         print(f'Epoch: {epoch+1:3d} | loss: {tloss:.4f}/{vloss:.4f}, tau: {tau:.1f}, Lip: {lipmin:.3f}/{lipmax:.2f}')

#     eloss = fitloss(model_state, model_state.params, data['xeval'], data['peval'], data['yeval'])
#     print(f'{name}: eval loss: {eloss:.4f}')

#     data['train_loss'] = jnp.array(train_loss)
#     data['val_loss'] = jnp.array(val_loss)
#     data['lipmin'] = jnp.array(Lipmin)
#     data['lipmax'] = jnp.array(Lipmax)
#     data['tau'] = jnp.array(Tau)
#     data['eval_loss'] = eloss

#     scipy.io.savemat(f'{train_dir}/data.mat', data)

#     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     save_args = orbax_utils.save_args_from_target(model_state.params)
#     orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)
