from datetime import datetime
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from robustnn import ren_base as ren
from utils import l2_norm


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
        R = X[1:-1] * (1 - X[1:-1]) * (X[1:-1] - 0.5) # TODO: *0.5?
        laplacian = (X[:-2] + X[2:] - 2 * X[1:-1]) / dx**2
        
        Xn = Xn.at[1:-1].set(X[1:-1] + dt * (laplacian + R / 2))
        Xn = Xn.at[0].set(U)
        Xn = Xn.at[-1].set(U)
    
    return X


def measure(X, U):
    """Measure input value b(t) at endpoints and X(t) in the middle."""
    return jnp.hstack((U, X[..., X.shape[-1] // 2]))


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
    
    # Set up states and input (at boundaries)
    X = init_x_func((time_steps, nx))        # State array xi(t)
    U = init_u_func((time_steps, n_in))      # Input array b(t)
    
    # Random perturbations
    ws = 0.05 * jax.random.normal(jax.random.PRNGKey(seed), time_steps-1)
    
    # Simulate discretised PDE through time with Euler integration
    # Input is normally distributed but clamped to [0,1]
    for t in range(time_steps - 1):
        X = X.at[t + 1].set(dynamics(X[t], U[t]))
        u_next = U[t] + ws[t]
        u_next = jnp.clip(u_next, 0, 1)
        U = U.at[t + 1].set(u_next)
    
    return X, U


def batch_data(xn, xt, input_data, batches, seed):
    """Split observer data up in time chunks for batches and shuffle order."""
    
    # Split into batches
    xt = jnp.array_split(xt, batches)
    xn = jnp.array_split(xn, batches)
    input_data = jnp.array_split(input_data, batches)

    # Shuffle batches
    key = jax.random.PRNGKey(seed)
    shuffle_indices = jax.random.permutation(key, len(xt))
    xt = xt[shuffle_indices]
    xn = xn[shuffle_indices]
    input_data = input_data[shuffle_indices]
    return list(zip(xn, xt, input_data))


def train_observer(
    model: ren.RENBase, data, epochs=50, lr=1e-3, min_lr=1e-7, seed=0, verbose=True
):
    """Train a REN to be an observer.
    
    Args:
        model (RENBase): REN model to train.
        data (list): Training data in batches. Each element should be `(xn, xt, input_data)`.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        lr: Initial learning rate. Defaults to 1e-3.
        min_lr: Minimum learning rate after decay. Defaults to 1e-7.
        seed (int, optional): Default random seed. Defaults to 123.
        verbose (bool, optional): Whether to print. Defaults to True.
        
    Returns:
        params: Parameters of trained model.
        mean_loss (list): List of training losses for each epoch.
    """
    
    def loss_fn(params, xn, x, u):
        """Loss function is one-step ahead prediction error."""
        x_pred, _ = model.apply(params, x, u)
        return jnp.mean(l2_norm(xn - x_pred, axis=-1)**2)
    
    grad_loss = jax.jit(jax.value_and_grad(loss_fn))
    
    @jax.jit
    def train_step(params, opt_state, scheduler_state, xn, x, u):
        """Run a single SGD training step."""
        loss_value, grads = grad_loss(model, xn, x, u)
        updates, opt_state = optimizer.update(grads, opt_state)
        updates = otu.tree_scalar_mul(scheduler_state.scale, updates)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    # Random seeds
    rng = jax.random.PRNGKey(seed)
    key1, rng = jax.random.split(rng)
    
    # Set up the optimizer with a learning rate scheduler that decays by 0.1
    # every time the mean training loss increases
    optimizer = optax.adam(lr)
    scheduler = optax.contrib.reduce_on_plateau(
        factor=0.1,
        min_scale=min_lr,
        patience=1          # Decay if no improvement after this many steps
    )
    
    # Initialise the REN and optimizer/scheduler states
    init_x = data[0][1]
    init_u = data[0][2]
    params = model.init(key1, init_x, init_u)
    opt_state = optimizer.init(params)
    scheduler_state = scheduler.init(params)
    
    # Loop through for training
    mean_loss, loss_std = [1e5], []
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
        
        # Print results for the user
        if verbose:
            current_lr = lr * scheduler_state.scale
            print("--------------------------------------------------------------------")
            print(f"Epoch: {epoch + 1:2d}\t " +
                  f"mean loss: {mean_loss[-1]:.4E}\t " +
                  f"std: {jnp.std(jnp.array(batch_loss)):.4E}\t" +
                  f"lr: {current_lr:.2g}\t" +
                  f"Time: {datetime.now()}")
            print("--------------------------------------------------------------------")
        
        # Update the learning rate scaling factor
        _, scheduler_state = scheduler.update(
            updates=params, state=scheduler_state, value=mean_loss
        )
    
    results = {"mean_loss": jnp.array(mean_loss), "std_loss": jnp.array(loss_std)}
    return params, results
