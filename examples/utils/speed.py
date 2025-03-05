import numpy as np

import pickle
from pathlib import Path

dirpath = Path(__file__).resolve().parent


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
