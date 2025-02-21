import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from pathlib import Path

from robustnn import ren
from utils.plot_utils import startup_plotting
from utils import data_handling as handler
from utils import utils_sysid as utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")

# Set up random seed
seed = 0
rng = jax.random.PRNGKey(seed)

# Problem setup
nx = 51             # Number of states
n_in = 1            # Number of inputs
L = 10.0            # Size of spatial domain
sigma = 0.1         # Used to construct time step

# Discretise space and time
dx = L / (nx - 1)
dt = sigma * dx**2

def dynamics(X0, U, steps=5):
    """Evaluate discretised dynamics: RHS of PDE.
    
    TODO: U is "b(t)" from the paper.
    TODO: X is the state "xi" from the paper.
    """
    X, Xn = X0, X0
    for _ in range(steps): # TODO: Why iterate?
        X = Xn
        
        def R(v):
            return v[1:-1] * (1 - v[1:-1]) * (v[1:-1] - 0.5) # TODO: *0.5?
        
        def laplacian(v):
            return (v[:-2] + v[2:] - 2 * v[1:-1]) / dx**2
        
        Xn = Xn.at[1:-1].set(X[1:-1] + dt * (laplacian(X) + R(X) / 2))
        Xn = Xn.at[0].set(U)
        Xn = Xn.at[-1].set(U)
    
    return X


def measure(X, U):
    """Measure input value b(t) at endpoints and X(t) in the middle."""
    return jnp.hstack((U, X[..., X.shape[-1] // 2]))


def get_data(time_steps=1000, init_func=jnp.zeros, seed=0):
    """Compute PDE state/input data through time."""
    
    # Set up states and input (at boundaries)
    X = init_func((time_steps, nx))        # State array xi(t)
    U = init_func((time_steps, n_in))      # Input array b(t)
    
    # Random perturbations
    ws = 0.05 * jax.random.normal(jax.random.PRNGKey(seed), time_steps-1)
    
    # TODO: Re-write with jax.lax.scan
    for t in range(time_steps - 1):
        X = X.at[t + 1].set(dynamics(X[t], U[t]))
        u_next = U[t] + ws[t]
        u_next = jnp.clip(u_next, 0, 1)
        U = U.at[t + 1].set(u_next)
    
    return X, U

# Get simulated data
X, U = get_data(100_000)
xt = X[:-1]                 # X at time t
xn = X[1:]                  # X at time t+1
y = measure(X, U)           # Measured end points and middle
input_data = y[:-1]

# Split the data into batches
batches = 200
xt = jnp.array_split(xt, batches)
xn = jnp.array_split(xn, batches)
input_data = jnp.array_split(input_data, batches)

# Shuffle batches
key, rng = jax.random.split(rng)
shuffle_indices = jax.random.permutation(key, len(xt))
xt = xt[shuffle_indices]
xn = xn[shuffle_indices]
input_data = input_data[shuffle_indices]
data = list(zip(xn, xt, input_data))

# Create the REN
nv = 500
nu = input_data.shape[-1]
ny = nx
model = ren.ContractingREN(
    nu, nx, nv, ny,
    activation=nn.tanh,
    do_polar_param=False,
    init_method="cholesky",
    # TODO: Remove output layer for observer example!!
    # It's not actually used and shouldn't be trainable.
)

def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.sum(x**2, **kwargs) + eps)


def train_observer(model: ren.RENBase, data, epochs=50, lr=1e-3, min_lr=1e-7, seed=0, verbose=True):
    """Train a REN to be an observer.
    
    TODO: Add documentation!!
    """
    
    def loss_fn(params, xn, x, u):
        """Loss function is one-step ahead prediction error."""
        x_pred, _ = model.apply(params, x, u)
        return jnp.mean(l2_norm(xn - x_pred, axis=-1)**2)
    
    grad_loss = jax.jit(jax.value_and_grad(loss_fn))
    
    @jax.jit
    def train_step(params, opt_state, xn, x, u):
        """Run a single SGD training step"""
        loss_value, grads = grad_loss(model, xn, x, u)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    # Random seeds
    rng = jax.random.PRNGKey(seed)
    key1, rng = jax.random.split(rng)
    
    # Set up the optimizer
    # TODO: Add a scheduler
    optimizer = optax.adam(lr)
    
    # Initialise the REN and optimizer state
    init_x = data[0][1]
    init_u = data[0][2]
    params = model.init(key1, init_x, init_u)
    opt_state = optimizer.init(params)
    
    # Loop through for training
    mean_loss, loss_std = [1e5], []
    for epoch in range(epochs):
        
        # Compute batch losses
        batch_loss = []
        for xn_k, x_k, u_k in data:
            params, opt_state, loss_value = train_step(
                params, opt_state, xn_k, x_k, u_k
            )
            batch_loss.append(loss_value)
        
        # Store the losses
        losses = jnp.array(batch_loss)
        mean_loss.append(jnp.mean(losses))
        loss_std.append(jnp.std(losses))
        
        # Print results for the user
        if verbose:
            print("--------------------------------------------------------------------")
            print(f"Epoch: {epoch + 1:2d}\t " +
                  f"mean loss: {mean_loss[-1]:.4E}\t " +
                  f"std: {jnp.std(jnp.array(batch_loss)):.4E}")
            print("--------------------------------------------------------------------")
        
        # TODO: This is incorrect I suspect. Follow this guide:
        # https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html
        # https://optax.readthedocs.io/en/latest/_collections/examples/contrib/reduce_on_plateau.html
        if mean_loss[-1] >= mean_loss[-2]:
            lr = max(lr * 0.1, min_lr)
            opt_state = optax.scale_by_schedule(lambda _: lr).init(opt_state)
    
    return mean_loss, loss_std


# TODO: Train a model.

# TODO: Set up testing for observer.

# Plotting function
def plot_heatmap(data, title, xlabel="Time steps", ylabel=""):
    plt.figure(figsize=(6, 4))
    plt.imshow(data.T, cmap='inferno', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Example plotting
# plot_heatmap(X, "True State")
