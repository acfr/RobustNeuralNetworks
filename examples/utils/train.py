import flax.linen as linen
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
from pathlib import Path

dirpath = Path(__file__).resolve().parent

def get_activation(s: str):
    """Get activation function from flax.linen via string."""
    if s == "identity":
        return (lambda x: x)
    return eval("linen." + s)


def get_mse(y_true, y_pred):
    """Compute mean square error."""
    return jnp.mean(jnp.square(y_true - y_pred))


def generate_fname(config):
    """Generate a common file name for results loading/saving."""
    filename = "{}_nx{}_nv{}_{}_{}_s{}.pickle".format(
        config["experiment"],
        config["nx"],
        config["nv"],
        config["activation"],
        config["init_method"],
        config["seed"]
    )
    
    filepath = dirpath / f"../../results/{config['experiment']}/"
    if not filepath.exists():
        filepath.mkdir(parents=True)
        
    return filepath / filename


def save_results(config, params, results):
    """Save results from sysID experiments."""
    filepath = generate_fname(config)
    data = (config, params, results)
    with filepath.open('wb') as fout:
        pickle.dump(data, fout)
        
def load_results(filepath):
    """Load results from sysID experiments"""
    with filepath.open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)


def load_results_from_config(config):
    """Short-cut to load from config dictionary."""
    filepath = generate_fname(config)
    return load_results(filepath)


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
        optax.clip(config["clip_grad"]), #TODO: By global norm...?
        optax.inject_hyperparams(optax.adam)(learning_rate=scheduler)
    )
    return optimizer
    
    
def train(train_data, model, optimizer, epochs=200, seed=123, verbose=True):
    """Train model for system identification.

    Args:
        train_data (list): List of tuples (u,y) with training data arrays.
        model (RENBase): REN model to train.
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
        Loss function.
        
        Computes MSE and returns updated model state.
        """
        new_x, y_pred = model.apply(params, x, u)
        return jnp.mean(jnp.square(y - y_pred)), new_x
    
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
    rng = jax.random.PRNGKey(seed)
    key1, key2, rng = jax.random.split(rng, 3)

    # Initialize model parameters and optimizer state
    init_u = train_data[0][0]
    init_x = model.initialize_carry(key1, init_u.shape)
    params = model.init(key2, init_x, init_u)
    opt_state = optimizer.init(params)
    
    # Loop through for training
    train_loss_log = []
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
        lr = opt_state[1].hyperparams['learning_rate']
        
        if verbose:
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, lr: {lr:.3g}")
    
    return params, jnp.squeeze(jnp.vstack(train_loss_log))


def validate(model, params, val_data, washout=100, seed=123):
    """Test SysID model on validation set(s).

    Args:
        model (RENBase): REN model for system identification
        params: Parameters of trained model.
        val_data (list): List of tuples (u,y) with validation data arrays.
        washout (int, optional): Ignore the first few time-steps. Defaults to 100.
        seed (int, optional): Default random seed. Defaults to 123.

    Returns:
        dict: Dictionary of results.
    """

    results = []
    rng = jax.random.PRNGKey(seed)
    key, rng = jax.random.split(rng)
    
    # Allow for multiple validation sets (F16 only has 1 though, so loop has 1 iter)
    for u_val, y_val in val_data:
        
        # Compute model prediction
        key, rng = jax.random.split(rng)
        x = model.initialize_carry(key, u_val.shape)
        _, y_pred = model.apply(params, x, u_val)
        
        # Compute metrics
        mse = get_mse(y_val[washout:], y_pred[washout:])
        mean_y = jnp.mean(y_val)
        nrmse = jnp.sqrt(mse / get_mse(y_val[washout:], mean_y))
        
        # Store results
        results.append({
            "u": u_val, 
            "y": y_val, 
            "y_pred": y_pred, 
            "mse": mse,
            "nrmse": nrmse, 
            "washout": washout
        })
    
    # Convert to dict of arrays
    return {key: np.array([d[key] for d in results]) for key in results[0]}
