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
