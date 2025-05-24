import jax, jax.numpy as jnp
import optax

from robustnn.utils import l2_norm
from robustnn import ren_base as ren
from robustnn import r2dn


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
    key = jax.random.key(seed)
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


def compute_p_contractingren(model: ren.RENBase, ps: dict):
    
    p = ps["params"]["p"]
    X = ps["params"]["X"]
    Y1 = ps["params"]["Y1"]
    
    nx = model.state_size
    nv = model.features
    abar = model.abar
    
    H = model._x_to_h_contracting(X, p)
    H11 = H[:nx, :nx]
    H33 = H[(nv+nx):(2*nx+nv), (nv+nx):(2*nx+nv)]
    
    P_imp = H33
    E = (H11 + P_imp / abar**2 + Y1 - Y1.T)/2
    
    return E.T @ jnp.linalg.solve(P_imp, E)


def compute_p_contractingr2dn(model: r2dn.ContractingR2DN, ps: dict):
    
    p = ps["params"]["p"]
    X = ps["params"]["X"]
    Y = ps["params"]["Y"]
    B1 = ps["params"]["B1"]
    C1 = ps["params"]["C1"]
    
    nx = model.state_size
    H = model._x_to_h_contracting(X, p, B1, C1)
    H11 = H[:nx, :nx]
    H22 = H[nx:, nx:]
    
    E = (H11 + H22 + Y - Y.T) / 2
    P_imp = H22
    
    return E.T @ jnp.linalg.solve(P_imp, E)
