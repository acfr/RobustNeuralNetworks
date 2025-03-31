import flax.linen as linen # This is used in eval()
import jax.numpy as jnp
import numpy as np
import scipy.signal as signal
import pickle

from pathlib import Path
dirpath = Path(__file__).resolve().parent


def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.sum(x**2, **kwargs) + eps)


def l1_norm(x, **kwargs):
    return jnp.sum(jnp.abs(x), **kwargs)


def list_to_dicts(data):
    return {key: np.array([d[key] for d in data]) for key in data[0]}


def get_activation(s: str):
    """Get activation function from flax.linen via string."""
    if s == "identity":
        return (lambda x: x)
    return eval("linen." + s)


def generate_fname(config):
    """Generate a common file name for results loading/saving."""
    is_r2dn = "nh" in config.keys()
    nh = f"nh{config['nh'][0]}_" if is_r2dn else ""
    layers = f"nl{len(config['nh'])}_" if is_r2dn else ""
    polar_label = "polar" if config["polar"] else "nopolar"
    
    filename = "{}_{}_nx{}_nv{}_{}{}{}_{}_{}_s{}".format(
        config["experiment"],
        config["network"],
        config["nx"],
        config["nv"],
        nh,
        layers,
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


def choose_lbdn_width(nu, nx, ny, nv_ren, nv_r2dn, n_layers):
    """Choose width of LBDN layers in R2DN so that
    number of params matches up with a REN of the same size.
    
    Assumes fixed hidden width in the LBDN of n_layers layers.
    Eg: hidden = (nh, ) * n_layers
    """
    
    # Difference between num. REN and num. S-REN (LTI) params
    diff = (
        (4*nx*nv_ren + nv_ren**2) + 
        nv_ren*(nu + ny + 1) - 
        nv_r2dn*(nu + ny + 2*nx + 1)
    )
    
    # Coefficients for width of LBDN layers
    n_layers -= 1
    a = (1 + 2*n_layers)
    b = 2*(nv_r2dn + n_layers + 1)
    c = (nv_r2dn**2 + 2*nv_r2dn + n_layers + 2) - diff
    
    # Solve quadratic equation
    nh = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return int(np.ceil(nh))


def add_state_space(sys1, sys2):
    """Adds two discrete-time state-space models."""
    if sys1.dt != sys2.dt:
        raise ValueError("Systems must have the same sampling time to be added.")
    
    A = np.block([
        [sys1.A, np.zeros((sys1.A.shape[0], sys2.A.shape[1]))],
        [np.zeros((sys2.A.shape[0], sys1.A.shape[1])), sys2.A]
    ])
    B = np.vstack([sys1.B, sys2.B])
    C = np.hstack([sys1.C, sys2.C])
    D = sys1.D + sys2.D

    return signal.StateSpace(A, B, C, D, dt=sys1.dt)


def laguerre_terms(k: int, a: float, dt=None):
    """Build state-space versions of Laguerre terms."""
    A = -a * np.eye(k) + np.triu(-2*a * np.ones((k, k)), 1)
    A = np.block([
        [A, 2*a * np.ones((k, 1))],
        [np.zeros((1, k)), -a]
    ])
    B = np.sqrt(2*a) * np.vstack([np.zeros((k, 1)), np.ones((1,1))])
    C = np.hstack([-np.ones((1, k)), np.ones((1,1))])
    D = np.zeros((1,1))
    
    sys = signal.StateSpace(A, B, C, D)
    sys = sys.to_discrete(dt, method='bilinear')
    return sys


def laguerre_composition(eigs, kmax=1, dt=1):
    """Compose a bunch of kth order (discrete) Laguerre terms
    for a sequence of interesting frequencies in `eigs`.
    """
    sys = laguerre_terms(kmax, eigs[0], dt)
    for k in range(1, len(eigs)):
        sys_k = laguerre_terms(kmax, eigs[k], dt)
        sys = add_state_space(sys, sys_k)
    return sys.A, sys.B, sys.C, sys.D
