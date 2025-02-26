import jax, jax.numpy as jnp
import optax

from robustnn.utils import l2_norm

def count_num_params(d):
    """
    Recursively counts the total number of elements in all jax.numpy arrays
    contained in a dictionary (which may contain nested dictionaries).
    
    Parameters:
    d (dict): Dictionary containing jax.numpy arrays and possibly nested dictionaries.
    
    Returns:
    int: Total number of elements in all jax.numpy arrays.
    """
    total_elements = 0
    for value in d.values():
        if isinstance(value, jnp.ndarray):
            total_elements += value.size
        elif isinstance(value, dict):
            total_elements += count_num_params(value)
    
    return total_elements


def estimate_lipschitz_lower(    
    policy,
    n_inputs,
    batches=128,
    max_iter=450,
    learning_rate=0.01,
    clip_at=0.01,
    init_var=0.001,
    verbose=True,
    seed=0
):
    """
    Estimate a lower-bound on the Lipschitz constant with gradient descent.
    
    Assumes "policy" is the model with syntax y = policy(u)
    """
    
    # Initialise model inputs
    key = jax.random.PRNGKey(seed)
    key, rng1, rng2 = jax.random.split(key, 3)
    u1 = init_var * jax.random.normal(rng1, (batches, n_inputs))
    u2 = u1 + 1e-4 * jax.random.normal(rng2, (batches, n_inputs))

    # Set up optimization parameters
    params = (u1, u2)

    # Optimizer
    scheduler = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=150,
        decay_rate=0.1,
        end_value=0.001*learning_rate,
        staircase=True
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_at),
        optax.inject_hyperparams(optax.adam)(learning_rate=scheduler),
        optax.scale(-1.0) # To maximise the Lipschitz bound
    )
    
    optimizer_state = optimizer.init(params)

    # Loss function
    def lip_loss(params, key):
        u1, u2 = params
        y1 = policy(u1)
        y2 = policy(u2)
        gamma = l2_norm(y2 - y1) / l2_norm(u1 - u2) # Can be numerical issues here!!
        return gamma

    # Gradient of the loss function
    grad_loss = jax.grad(lip_loss)
    jit_lip_loss = jax.jit(lip_loss)
    jit_grad_loss = jax.jit(grad_loss)

    # Use gradient descent to estimate the Lipschitz bound
    lips = []
    for iter in range(max_iter):
        
        key, rng1, rng2 = jax.random.split(key, 3)
        grad_value = jit_grad_loss(params, rng1)
        updates, optimizer_state = optimizer.update(grad_value, optimizer_state)
        params = optax.apply_updates(params, updates)
        
        lips.append(jit_lip_loss(params, rng2))
        if verbose and iter % 20 == 0:
            print("Iter: ", iter, "\t L: ", lips[-1], "\t lr: ", 
                  optimizer_state[1].hyperparams['learning_rate'])
    
    return max(lips)