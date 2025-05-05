#!/usr/bin/env python3
#
import jax
import jax.numpy as jnp
import jax.random as random
from typing import Sequence, Union

def Sampler(
        rng: random.PRNGKey,
        batches: int,
        data_dim: int,
        x_min: Union[float, jnp.ndarray] = -1.38,
        x_max: Union[float, jnp.ndarray] = 1.38,
):

    return random.uniform(rng, (batches, data_dim), minval=x_min, maxval=x_max)


def data_gen(
    rng: random.PRNGKey,
    data_dim: int = 20,
    # val_min: float = -2.*jnp.pi,
    # val_max: float = 2.*jnp.pi,
    val_min: float = -1.5, # Was 0.78
    val_max: float = 1.5,
    train_batch_size: int = 200,
    test_batch_size: int = 5000,
    train_batches: int = 200,
    test_batches: int = 1,
    eval_batch_size: int = 5000,
    eval_batches: int = 100,
):
    rng_train, rng_test, rng_eval = random.split(rng, 3)

    xtrain = Sampler(rng_train, train_batch_size * train_batches, data_dim, x_min=val_min, x_max=val_max)
    print(f"xtrain dim: {xtrain.shape}")

    xtest  = Sampler(rng_test, test_batch_size * test_batches, data_dim, x_min=val_min, x_max=val_max)

    xeval  = Sampler(rng_eval, eval_batch_size * eval_batches, data_dim, x_min=val_min, x_max=val_max)


    ytrain, ytest, yeval = TwoDoFArmField(xtrain), TwoDoFArmField(xtest), TwoDoFArmField(xeval)

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


def TwoDoFArmField(
    th_samples : jax.Array,
    # th1_range: Sequence[float] = [-2*jnp.pi, 2*jnp.pi],
    # th2_range: Sequence[float] = [-2*jnp.pi, 2*jnp.pi],
    # n_grid: int = 200,
    link_lengths: Sequence[float] = [0.5, 1.0]
):

    # print(f"recieved array: {th_samples}")
    th1 = th_samples[:,0]
    th2 = th_samples[:,1]
    # print(f"Creaing twodofarfield: th1: {th1_range}, th2: {th2_range}")
    # Create a meshgrid of joint angles
    # th1 = jnp.linspace(th1_range[0], th1_range[1], n_grid)
    # th2 = jnp.linspace(th2_range[0], th2_range[1], n_grid)
    # th1_grid, th2_grid = jnp.meshgrid(th1, th2)
    # print(f"th1_grid: {th1_grid.shape}")
    # print(f"th2_grid: {th2_grid.shape}")

    th1_flat = th1
    th2_flat = th2
    print(f"th1_flat: {th1}")
    print(f"th2_flat: {th2}")

    l1, l2 = link_lengths

    # L1 X/Y
    x1 = l1 * jnp.cos(th1_flat)
    y1 = l1 * jnp.sin(th1_flat)

    # EEF X/Y
    x2 = x1 + l2 * jnp.cos(th1_flat + th2_flat)
    y2 = y1 + l2 * jnp.sin(th1_flat + th2_flat)

    # Configuration space
    config_space = jnp.stack([th1_flat, th2_flat], axis=1)

    # Task space (end effector position)
    task_space = jnp.stack([x2, y2], axis=1)

    # return config_space, task_space, th1_grid, th2_grid
    return task_space
