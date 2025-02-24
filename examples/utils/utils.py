import flax.linen as linen
import jax.numpy as jnp
import pickle

from pathlib import Path
dirpath = Path(__file__).resolve().parent


def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.sum(x**2, **kwargs) + eps)


def l1_norm(x, **kwargs):
    return jnp.sum(jnp.abs(x), **kwargs)


def get_activation(s: str):
    """Get activation function from flax.linen via string."""
    if s == "identity":
        return (lambda x: x)
    return eval("linen." + s)


def generate_fname(config):
    """Generate a common file name for results loading/saving."""
    polar_label = "polar" if config["polar"] else "nopolar"
    filename = "{}_nx{}_nv{}_{}_{}_{}_s{}".format(
        config["experiment"],
        config["nx"],
        config["nv"],
        config["activation"],
        config["init_method"],
        polar_label,
        config["seed"]
    )
    
    filepath = dirpath / f"../../results/{config['experiment']}/"
    if not filepath.exists():
        filepath.mkdir(parents=True)
        
    return filepath / f"{filename}.pickle", filename


def save_results(config, params, results):
    """Save results from experiments"""
    filepath, _ = generate_fname(config)
    data = (config, params, results)
    with filepath.open('wb') as fout:
        pickle.dump(data, fout)
        
        
def load_results(filepath):
    """Load results from experiments"""
    with filepath.open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)


def load_results_from_config(config):
    """Short-cut to load from config dictionary."""
    filepath, _ = generate_fname(config)
    return load_results(filepath)
