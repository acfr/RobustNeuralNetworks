# DYS inverse tests

This folder isolates regression tests for the Davis-Yin inverse used by
regular and parameter-conditioned MonLip/BiLip networks. The tests cover the
BiLip cores used by PLNet and PPLNet in JAX and Torch, including nonzero hidden
biases, unsafe requested steps, adaptive convergence, depth, and distortion.

Run the suite with:

```bash
PYTHONPATH=. conda run -n plnet_mjx python test/dys_inverse/test_bilipnet_jax_inverse.py
PYTHONPATH=. conda run -n plnet_mjx python test/dys_inverse/test_jax_inverse_paths.py
PYTHONPATH=. conda run -n plnet_mjx python test/dys_inverse/test_torch_inverse_paths.py
```
