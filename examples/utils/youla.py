import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import numpy as np
from datetime import datetime
from scipy import signal

from robustnn import ren_base as ren
from .utils import l2_norm, l1_norm


class ExampleSystem:
    """Basic implementation of linear Youla system:
    
    [z; y~] = [T0 T1; T2 0] * [w; u~]
    
    where T0 = T1 = -T2 = tf([0.3], [1, -1.6cos(0.2pi), 0.64], dt=1.0).
    Inputs u~ are clamped between [-5, 5].
    """
    
    def __init__(self):
        
        # Linear system 
        rho = 0.8
        phi = 0.2*np.pi
        gain = 0.3
        den = [1, -2*rho*np.cos(phi), rho**2]
        T = signal.TransferFunction([gain], den, dt=1.0).to_ss()
        
        # Construct the Youla system (z, y~) = [T T; -T 0](w, u~)
        A, B, C = jnp.array(T.A), jnp.array(T.B), jnp.array(T.C)
        A_zero = jnp.zeros(A.shape)
        B_zero = jnp.zeros(B.shape)
        C_zero = jnp.zeros(C.shape)
        
        self.A = jnp.block([
            [A, A_zero, A_zero],
            [A_zero, A, A_zero],
            [A_zero, A_zero, A],
        ])
        self.B = jnp.block([[B_zero], [B], [B_zero]])
        self.Bw = jnp.block([[B], [B_zero], [B]])
        self.C = jnp.block([
            [C, C, C_zero],
            [C_zero, C_zero, -C],
        ])
        
        # Some useful data to store
        self.nx = self.A.shape[0]       # Number of states
        self.nu = self.B.shape[1]       # Number of inputs
        self.ny = self.C.shape[0]       # Number of outputs
        self.nz = C.shape[0]            # Dimension of z (performance output)
        self.max_u = 5                  # Control limits

    def init_state(self, batches:int = 1):
        """Generate initial state of zeros."""
        return jnp.zeros((batches, self.nx))
    
    def dynamics(self, x, w, u):
        """Step dynamics by one timestep with clipped control inputs."""
        u_clamped = jnp.clip(u, min=-self.max_u, max=self.max_u)
        return x @ self.A.T + w @ self.Bw.T + u_clamped @ self.B.T
    
    def measure(self, x):
        """Get measurement output."""
        return x @ self.C.T
    
    
def generate_disturbance(
    key, 
    timesteps: int, 
    batches: int, 
    nw: int = 1, 
    hold_time: int = 50, 
    amp: int = 10, 
    n_segments: int = 1
):
    """Generate a batch of disturbance sequences over a given time interval.
    
    The disturbances are piecewise constant with a specified hold time and a magnitude 
    uniformly distributed in the interval [-amp, amp]. The output can be split into 
    multiple segments if needed.
    
    Args:
        key (jax.random.PRNGKey): Random key for JAX's random number generator.
        timesteps (int): Total number of timesteps in the sequence.
        batches (int): Number of batch samples to generate.
        nw (int, optional): Number of disturbance channels (default: 1).
        hold_time (int, optional): Number of timesteps each disturbance value is held constant (default: 50).
        amp (int, optional): Maximum absolute magnitude of disturbance values (default: 10).
        n_segments (int, optional): Number of segments to split the output into (default: 1).
        
    Returns:
        jnp.ndarray or list of jnp.ndarray: A JAX array of shape (timesteps, batches, nw) 
        if n_segments is 1, otherwise a list of JAX arrays split into n_segments.
    """
    rng, key = jax.random.split(key)
    num_pieces = int(jnp.ceil(timesteps / hold_time))
    disturbances = jax.random.uniform(
        rng, 
        minval=-amp, 
        maxval=amp,
        shape=(num_pieces, batches, nw),
    )
    disturbances = jnp.repeat(disturbances, hold_time, axis=0)
    
    
    # Truncate to match the required timesteps, return in chunks
    d = disturbances[:timesteps]
    return d if n_segments == 1 else jnp.array_split(d, n_segments)
    

def rollout(
    env: ExampleSystem, 
    model: ren.RENBase,
    params: dict,
    env_state: jnp.ndarray,
    ren_state: jnp.ndarray,
    disturbances: jnp.ndarray,
    ):
    """Roll out the closed-loop system.
    
    Rolls out for as many timesteps as in the leading dimension
    of disturbances.
    """
    
    # Construct the explicit params from the current REN params
    # Don't need to do this every step during the rollout!
    explicit = model.direct_to_explicit(params)
    
    def youla_step(carry, wt):
        """Single timestep of closed-loop system."""
        xt, qt = carry
        
        # Get measurements
        env_out = env.measure(xt)
        z = env_out[..., :env.nz]
        y_tilde = env_out[..., env.nz:]
        
        # Compute controls with Youl param
        # We clip states here just for convenience in plotting/costs, 
        # they're already clipped in the dynamics() function
        q_next, u_tilde = model.explicit_call(params, qt, y_tilde, explicit)
        u_tilde = jnp.clip(u_tilde, min=-env.max_u, max=env.max_u)
        
        # Update environment state and return
        x_next = env.dynamics(xt, wt, u_tilde)
        return (x_next, q_next), (z, u_tilde)
    
    init_carry = (env_state, ren_state)
    return jax.lax.scan(youla_step, init_carry, disturbances)
    

def train_yoularen(
    env: ExampleSystem, 
    model: ren.RENBase, 
    lr = 1e-3, 
    min_lr = 1e-6,
    lr_patience = 10,
    epochs: int = 100, 
    batches: int = 32,
    test_batches: int = 32,
    rollout_length: int = 100, 
    max_steps: int = 200, 
    verbose: bool = True,
    seed: int = 0,
):
    """Train a Youla-REN with analytic policy gradients.

    Args:
        env (ExampleSystem): Linear system to control.
        model (ren.RENBase): REN model to train.
        lr (optional): Initial learning rate. Defaults to 1e-3.
        min_lr: Minimum learning rate after decay. Defaults to 1e-7.
        lr_patience: How many steps loss can increase before decay imposed. Defaults to 1.
        epochs (int, optional):  Number of training epochs. Defaults to 100.
        batches (int, optional):  Number of training batches. Defaults to 32.
        test_batches (int, optional):  Number of test batches. Defaults to 32.
        rollout_length (int, optional):  Number of timesteps per epoch. Defaults to 100.
        max_steps (int, optional):  Number of timesteps before reset. Defaults to 200.
            Must be integer multiple of rollout_length.
        verbose (bool, optional): Whether to print. Defaults to True.
        seed (int, optional): Default random seed. Defaults to 0.
        
    Returns:
        params: Parameters of trained model.
        results (dict): Dictionary of training losses (mean, std).
    """
    
    @jax.jit
    def loss_fn(params, x, q, w):
        """Loss function rolls out the policy for some time."""
        (x_next, q_next), (z, u_tilde) = rollout(env, model, params, x, q, w)
        loss_z = l1_norm(z, axis=-1)
        loss_u = 1e-4 * l2_norm(u_tilde, axis=-1)**2
        loss = jnp.mean(loss_z + loss_u)
        return loss, (x_next, q_next)
    
    grad_loss = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    
    @jax.jit
    def train_step(params, opt_state, scheduler_state, x, q, w):
        """Run a single SGD training step."""
        (loss_value, states), grads = grad_loss(params, x, q, w)
        x_next, q_next = states
        updates, opt_state = optimizer.update(grads, opt_state)
        updates = otu.tree_scalar_mul(scheduler_state.scale, updates)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, x_next, q_next
    
    # Only support max_steps as integer multiple of rollout_length
    # for now. Can easily change this later
    assert max_steps % rollout_length == 0
    num_epochs_per_reset = max_steps // rollout_length
    
    # Random seeds
    rng = jax.random.PRNGKey(seed)
    key1, key2, key3, key4, rng = jax.random.split(rng, 5)
    
    # Set up optimizer and learning rate scheduler
    optimizer = optax.adam(lr)
    scheduler = optax.contrib.reduce_on_plateau(
        factor=0.1,
        min_scale=min_lr / lr,
        patience=lr_patience        # Decay if no improvement after this many steps
    )
    
    # Initialise the REN and optimizer
    env_state = env.init_state(batches)
    y_tilde = env.measure(env_state)[..., env.nz:]
    input_shape = y_tilde.shape
    
    ren_state = model.initialize_carry(key1, input_shape)
    params = model.init(key2, ren_state, y_tilde)
    opt_state = optimizer.init(params)
    scheduler_state = scheduler.init(params)
    
    # Test dataset
    test_x0 = env.init_state(test_batches)
    test_q0 = model.initialize_carry(key4, (test_batches, y_tilde.shape[-1]))
    test_disturbances = generate_disturbance(key3, max_steps, batches)
    
    # Loop through for training
    test_loss = []
    train_loss = []
    timelog = []
    for epoch in range(epochs):
        
        # Evaluate test loss for logging
        timelog.append(datetime.now())
        test_loss.append(
            loss_fn(params, test_x0, test_q0, test_disturbances)[0]
        )
            
        # Reset the environment, policy states,
        key1, key2, rng = jax.random.split(rng, 3)
        env_state = env.init_state(batches)
        ren_state = model.initialize_carry(key1, input_shape)
        
        # Generate new batch of disturbances and split into segments
        disturbances = generate_disturbance(
            key2, max_steps, batches, n_segments=num_epochs_per_reset
        )
    
        # Train over rollout_length segments and store losses for full rollout
        batch_loss = []
        for k in range(num_epochs_per_reset):
            params, opt_state, loss_value, env_state, ren_state = train_step(
                params,
                opt_state,
                scheduler_state,
                env_state,
                ren_state,
                disturbances[k]
            )
            batch_loss.append(loss_value)
            
        # Store loss over full rollout and print training info
        train_loss.append(jnp.array(batch_loss).mean())
        current_lr = lr * scheduler_state.scale
        
        if verbose:
            print(f"epoch: {epoch+1}/{epochs}, " +
                  f"train_cost: {train_loss[-1]:.4f}, " +
                  f"test_cost: {test_loss[-1]:.4f}, " +
                  f"lr: {current_lr:.3g}, " +
                  f"Time: {timelog[-1]}")
            
        # Update the learning rate scaling factor
        _, scheduler_state = scheduler.update(
            updates=params, state=scheduler_state, value=test_loss[-1]
        )
    
    # Final test cost, store results, return
    timelog.append(datetime.now())
    test_loss.append(loss_fn(params, test_x0, test_q0, test_disturbances)[0])
    results = {
        "train_loss": jnp.array(train_loss), 
        "test_loss": jnp.array(test_loss),
        "times": timelog,
    }
    return params, results
