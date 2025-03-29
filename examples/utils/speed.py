import numpy as np

import pickle
from pathlib import Path

dirpath = Path(__file__).resolve().parent


def list_to_dicts(data):
    return {key: np.array([d[key] for d in data]) for key in data[0]}


def generate_filepath(filename):
    filepath = dirpath / f"../../results/timing/"
    if not filepath.exists():
        filepath.mkdir(parents=True)
    return filepath / f"{filename}.pickle"


def save_results(fname, results):
    """Save results from experiments"""
    filepath = generate_filepath(fname)
    data = (results,)
    with filepath.open('wb') as fout:
        pickle.dump(data, fout)
        
        
def load_results(filepath):
    """Load results from experiments"""
    with filepath.open('rb') as fin:
        buf = fin.read()
    data = pickle.loads(buf)
    return data[0]


def compute_num_ren_params(model):
    """Analytically compute number of REN params to save on computation."""
    nu = model.input_size
    nx = model.state_size
    nv = model.features
    ny = model.output_size
    d = min(nu, ny)
    return (
        nx * nu +
        nv * nu +
        nx + nv + 
        nx * nx +
        (2*nx + nv)**2 + 1 +
        ny +
        ny * nx +
        ny + nv +
        ny + nu +
        2*d * d +
        abs(ny - nu) * d
    )


def compute_num_r2dn_params(model):
    """
    Analytically compute number of R2DN params 
    to save on computation. Assumes all hidden
    layers have the same fixed width.
    """
    nu = model.input_size
    nx = model.state_size
    nv = model.features
    ny = model.output_size
    nh = model.hidden[0]
    n_layers = len(model.hidden)-1
    r2dn_ps = (
        nx * nu +
        nv * nu +
        nx + nv + 
        nx * nx +
        nx * nv +
        nv * nx +
        2*nx * 2*nx + 1 +
        ny +
        ny * nx +
        ny + nv +
        ny + nu
    )
    a = (1 + 2*n_layers)
    b = 2*(nv + n_layers + 1)
    c = (nv**2 + 2*nv + n_layers + 2)
    r2dn_ps += a * nh**2 + b * nh + c
    return r2dn_ps