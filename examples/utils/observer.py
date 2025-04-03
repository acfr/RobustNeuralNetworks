from datetime import datetime
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from robustnn import ren_base as ren
from .utils import l2_norm


def dynamics(X0, U, steps=5, L=10.0, sigma=0.1):
    """Evaluate discretised dynamics with Euler integration: RHS of PDE.
    
    U is "b(t)" from the REN paper.
    X is the state "xi" from the REN paper.
    """
    
    # Compute space/time discretisation
    nx = X0.shape[-1]
    dx = L / (nx - 1)
    dt = sigma * dx**2
    
    # Solve for X with discretised PDE
    Xn = X0
    for _ in range(steps):
        
        X = Xn
        R = X[1:-1] * (1 - X[1:-1]) * (X[1:-1] - 0.5)
        laplacian = (X[:-2] + X[2:] - 2 * X[1:-1]) / dx**2
        
        Xn = Xn.at[1:-1].set(X[1:-1] + dt * (laplacian + R / 2))
        Xn = Xn.at[:1].set(U)
        Xn = Xn.at[-1:].set(U)
    
    return X


def measure(X, U):
    """Measure input value b(t) at endpoints and X(t) in the middle."""
    indx_middle = X.shape[-1] // 2
    return jnp.hstack((U, X[..., indx_middle:indx_middle+1]))


def get_data(
    nx=51, 
    n_in=1, 
    time_steps=1000, 
    init_x_func=jnp.zeros,
    init_u_func=jnp.zeros,
    seed=0
):
    """Compute PDE state/input data through time.
    
    Option to initialise the states/inputs however you like.
    """
    
    # Initial states and inputs
    X0 = init_x_func((nx,))     # Initial state
    U0 = init_u_func((n_in,))   # Initial input
    
    # Random perturbations
    ws = 0.05 * jax.random.normal(jax.random.key(seed), (time_steps-1, n_in))
    
    # Simulate discretised PDE through time with Euler integration
    # Input is normally distributed but clamped to [0,1]
    def step(carry, w_t):
        X_t, U_t = carry
        X_next = dynamics(X_t, U_t)
        U_next = jnp.clip(U_t + w_t, 0, 1)
        return (X_next, U_next), (X_next, U_next)
    
    _, (X, U) = jax.lax.scan(step, (X0, U0), ws)
    
    return jnp.vstack([X0, X]), jnp.vstack([U0, U])


def batch_data(xn, xt, input_data, batches, seed):
    """Split observer data up in time chunks for batches and shuffle order."""
    
    # Split into batches
    xt = jnp.array_split(xt, batches)
    xn = jnp.array_split(xn, batches)
    input_data = jnp.array_split(input_data, batches)

    # Shuffle batches
    key = jax.random.key(seed)
    shuffle_indices = jax.random.permutation(key, len(xt))
    xt = [xt[i] for i in shuffle_indices]
    xn = [xn[i] for i in shuffle_indices]
    input_data = [input_data[i] for i in shuffle_indices]
    return list(zip(xn, xt, input_data))


def train_observer(
    model: ren.RENBase, data, epochs=50, lr=1e-3, min_lr=1e-7, seed=0, verbose=True,
    lr_patience=1
):
    """Train a REN to be an observer.
    
    Args:
        model (ren.RENBase): REN model to train.
        data (list): Training data in batches. Each element should be `(xn, xt, input_data)`.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        lr: Initial learning rate. Defaults to 1e-3.
        min_lr: Minimum learning rate after decay. Defaults to 1e-7.
        lr_patience: How many steps mean loss can increase before decay imposed. Defaults to 1.
        seed (int, optional): Default random seed. Defaults to 0.
        verbose (bool, optional): Whether to print. Defaults to True.
        
    Returns:
        params: Parameters of trained model.
        results (dict): Dictionary of training losses (mean, std).
    """
    
    def loss_fn(params, xn, x, u):
        """Loss function is one-step ahead prediction error."""
        x_pred, _ = model.apply(params, x, u)
        return jnp.mean(l2_norm(xn - x_pred, axis=-1)**2)
    
    grad_loss = jax.jit(jax.value_and_grad(loss_fn))
    
    @jax.jit
    def train_step(params, opt_state, scheduler_state, xn, x, u):
        """Run a single SGD training step."""
        loss_value, grads = grad_loss(params, xn, x, u)
        updates, opt_state = optimizer.update(grads, opt_state)
        updates = otu.tree_scalar_mul(scheduler_state.scale, updates)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    # Random seeds
    rng = jax.random.key(seed)
    key1, rng = jax.random.split(rng)
    
    # Set up the optimizer with a learning rate scheduler that decays by 0.1
    # every time the mean training loss increases
    optimizer = optax.adam(lr)
    scheduler = optax.contrib.reduce_on_plateau(
        factor=0.1,
        min_scale=min_lr / lr,
        patience=lr_patience        # Decay if no improvement after this many steps
    )
    
    # Initialise the REN and optimizer/scheduler states
    init_x = data[0][1]
    init_u = data[0][2]
    params = model.init(key1, init_x, init_u)
    opt_state = optimizer.init(params)
    scheduler_state = scheduler.init(params)
    
    # Loop through for training
    mean_loss, loss_std = [1e5], []
    timelog = []
    for epoch in range(epochs):
        
        # Compute batch losses
        batch_loss = []
        for xn_k, x_k, u_k in data:
            params, opt_state, loss_value = train_step(
                params, opt_state, scheduler_state, xn_k, x_k, u_k
            )
            batch_loss.append(loss_value)
        
        # Store the losses
        losses = jnp.array(batch_loss)
        mean_loss.append(jnp.mean(losses))
        loss_std.append(jnp.std(losses))
        timelog.append(datetime.now())
        
        # Print results for the user
        if verbose:
            current_lr = lr * scheduler_state.scale
            print(f"Epoch: {epoch + 1:2d}, " +
                  f"mean loss: {mean_loss[-1]:.4E}, " +
                  f"std: {jnp.std(jnp.array(batch_loss)):.4E}, " +
                  f"lr: {current_lr:.2g}, " +
                  f"Time: {timelog[-1]}")
        
        # Update the learning rate scaling factor
        _, scheduler_state = scheduler.update(
            updates=params, state=scheduler_state, value=mean_loss[-1]
        )
    
    results = {
        "mean_loss": jnp.array(mean_loss[1:]), 
        "std_loss": jnp.array(loss_std),
        "times": timelog,
    }
    return params, results


def validate(model: ren.RENBase, params, horizon=2000, seed=0):
    
    # Generate some test data
    def init_u_func(*args, **kwargs):
        return 0.5*jnp.ones(*args, **kwargs)
        
    xtrue, u = get_data(
        time_steps=horizon+1,
        init_u_func=init_u_func,
        init_x_func=jnp.ones,
        nx=model.state_size,
        seed=seed,
    )
    y = measure(xtrue, u)
    
    # Observer estimates through time
    key = jax.random.key(seed)
    x0 = model.initialize_carry(key, y[0].shape)
    _, xhat = model.simulate_sequence(params, x0, y)
    
    # Compute normalised root mean square error too as a fit metric
    mse = jnp.mean((xtrue - xhat)**2)
    nrmse = jnp.sqrt(mse / jnp.mean((xtrue)**2))
    
    return {
        "true_states": xtrue,
        "pred_states": xhat,
        "mse": mse,
        "nrmse": nrmse,
    }
