import flax.linen as linen
import jax.numpy as jnp
import numpy as np
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
    filename = "{}_{}_nx{}_nv{}_{}_{}_{}_s{}".format(
        config["experiment"],
        config["network"],
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


def choose_lbdn_width(nu, nx, ny, nv_ren, nv_sren, n_layers):
    """Choose width of LBDN layers in scalable REN so that
    number of params matches up with a REN of the same size.
    
    Assumes fixed hidden width in the LBDN of n_layers layers.
    Eg: hidden = (nh, ) * n_layers
    """
    
    # Difference between num. REN and num. S-REN (LTI) params
    diff = (
        (4*nx*nv_ren + nv_ren**2) + 
        nv_ren*(nu + ny + 1) - 
        nv_sren*(nu + ny + 2*nx + 1)
    )
    
    # Coefficients for width of LBDN layers
    n_layers -= 1
    a = (1 + 2*n_layers)
    b = 2*(nv_sren + n_layers + 1)
    c = (nv_sren**2 + 2*nv_sren + n_layers + 2) - diff
    
    # Solve quadratic equation
    nh = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return int(np.ceil(nh))
