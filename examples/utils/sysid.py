from datetime import datetime
import jax
import jax.numpy as jnp
import optax
from pathlib import Path

from robustnn import ren_base as ren
from .utils import l2_norm

dirpath = Path(__file__).resolve().parent


def setup_optimizer(config, n_segments):
    """Set up optimizer for training

    Args:
        config (dict): Training/model config options.
        n_segments (int): Number of segments in training data.
    """
    steps = config["schedule"]["decay_steps"] * n_segments
    scheduler = optax.exponential_decay(
        init_value=config["schedule"]["init_value"],
        transition_steps=steps,
        decay_rate=config["schedule"]["decay_rate"],
        end_value=config["schedule"]["end_value"],
        staircase=True
    )
    optimizer = optax.chain(
        optax.clip(config["clip_grad"]),
        optax.inject_hyperparams(optax.adam)(learning_rate=scheduler)
    )
    return optimizer

    
def train(train_data, model: ren.RENBase, optimizer, epochs=200, seed=123, verbose=True):
    """Train model for system identification.

    Args:
        train_data (list): List of tuples (u,y) with training data arrays.
        model (ren.RENBase): REN model to train.
        optimizer: Optimizer for training.
        epochs (int, optional): Number of training epochs. Defaults to 200.
        seed (int, optional): Default random seed. Defaults to 123.
        verbose (bool, optional): Whether to print. Defaults to True.
        
    Returns:
        params: Parameters of trained model.
        train_loss_log (list): List of training losses for each epoch.
    """
    
    def loss_fn(params, x, u, y):
        """
        Computes loss (l2 norm of simulation error) and returns
        updated model state. 
        
        Loss takes mean over time index only for consistency with
        original REN paper.
        """
        new_x, y_pred = model.simulate_sequence(params, x, u)
        loss = jnp.mean(l2_norm(y - y_pred, axis=(-2, -1))**2)
        return loss, new_x
    
    grad_loss = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    @jax.jit
    def train_step(params, opt_state, x, u, y):
        """
        Run a single training update step (SGD).
        """
        (loss_value, new_x), grads = grad_loss(params, x, u, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, new_x, loss_value
    
    # Random seeds
    rng = jax.random.key(seed)
    key1, key2, rng = jax.random.split(rng, 3)

    # Initialize model parameters and optimizer state
    init_u = train_data[0][0][0] # (u, sequence 0, time 0)
    init_x = model.initialize_carry(key1, init_u.shape)
    params = model.init(key2, init_x, init_u)
    opt_state = optimizer.init(params)
    
    # Loop through for training
    train_loss_log = []
    timelog = []
    for epoch in range(epochs):
        
        # Reset the recurrent state
        key, rng = jax.random.split(rng)
        x = model.initialize_carry(key, init_u.shape)
        
        # Compute batch loss
        batch_loss = []
        for u, y in train_data:
            params, opt_state, x, loss_value = train_step(
                params, opt_state, x, u, y
            )
            batch_loss.append(loss_value)

        # Store losses and print training info
        epoch_loss = jnp.mean(jnp.array(batch_loss))
        train_loss_log.append(epoch_loss)
        timelog.append(datetime.now())
        lr = opt_state[1].hyperparams['learning_rate']
        
        if verbose:
            print(f"Epoch: {epoch+1}/{epochs}, " +
                  f"Loss: {epoch_loss:.4f}, " +
                  f"lr: {lr:.3g}, " +
                  f"Time: {timelog[-1]}")
    results = {
        "train_loss": jnp.array(train_loss_log),
        "times": timelog,
    }
    return params, results


def validate(model: ren.RENBase, params, val_data, washout=100, seed=123):
    """Test SysID model on validation set(s).

    Args:
        model (ren.RENBase): REN model for system identification
        params: Parameters of trained model.
        val_data (tuple): Tuple (u,y) with validation data arrays.
        washout (int, optional): Ignore the first few time-steps. Defaults to 100.
        seed (int, optional): Default random seed. Defaults to 123.

    Returns:
        dict: Dictionary of results.
    """

    rng = jax.random.key(seed)
    key, rng = jax.random.split(rng)
    u_val, y_val = val_data
        
    # Compute model prediction
    key, rng = jax.random.split(rng)
    x0 = model.initialize_carry(key, u_val[0].shape)
    _, y_pred = model.simulate_sequence(params, x0, u_val)
    
    # Compute metrics
    y1, y2 = y_val[washout:], y_pred[washout:]
    mse = jnp.mean((y1 - y2)**2)
    nrmse = jnp.sqrt(mse / jnp.mean((y1)**2))
    
    # Return results
    return {
        "u": u_val, 
        "y": y_val, 
        "y_pred": y_pred, 
        "mse": mse,
        "nrmse": nrmse, 
        "washout": washout
    }
