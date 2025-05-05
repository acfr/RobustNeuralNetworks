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
    data_dim: int = 3,
    val_min: float = -1.5,
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
    ytrain, ytest, yeval = ThreeDoFArmField(xtrain), ThreeDoFArmField(xtest), ThreeDoFArmField(xeval)
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

def ThreeDoFArmField(
    th_samples: jax.Array,
    link_lengths: Sequence[float] = [0.5, 0.75, 1.0]  # Three links for 3-DoF
):
    # Extract the three joint angles
    th_x = th_samples[:, 0]  # Rotation around x-axis
    th_y = th_samples[:, 1]  # Rotation around y-axis
    th_z = th_samples[:, 2]  # Rotation around z-axis

    l1, l2, l3 = link_lengths

    x1 = 0
    y1 = l1 * jnp.cos(th_x)
    z1 = l1 * jnp.sin(th_x)

    x2 = x1 + l2 * jnp.sin(th_y)
    y2 = y1
    z2 = z1 + l2 * jnp.cos(th_y)

    x3 = x2
    y3 = y2 + l3 * jnp.sin(th_z)
    z3 = z2 + l3 * jnp.cos(th_z)

    # Configuration space (joint angles)
    config_space = jnp.stack([th_x, th_y, th_z], axis=1)

    # Task space (end effector position in 3D)
    task_space = jnp.stack([x3, y3, z3], axis=1)

    return task_space
