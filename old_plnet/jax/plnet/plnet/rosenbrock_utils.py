import jax
import jax.numpy as jnp 
import jax.random as random
from typing import Sequence, Union

def Rosenbrock(x):
    f = lambda x, y : (x-1.) ** 2 / 200. + 0.5 * (y - x ** 2) ** 2

    single = x.ndim == 1
    if single:
        x = jnp.expand_dims(x, 0)

    N = jnp.shape(x)[-1]
    y = jnp.stack([f(x[..., i], x[..., i+1]) for i in range(N-1)], axis=1)
    y = jnp.mean(y, axis=1)

    if single:
        y = jnp.squeeze(y, 0)
    
    return y 

def PRosenbrock(x, p):
    f = lambda x, y, a, b: (x- a) ** 2 / 200. + 0.5 * (y - b * x ** 2) ** 2

    single = x.ndim == 1
    if single:
        x = jnp.expand_dims(x, 0)
        p = jnp.expand_dims(p, 0)

    N = jnp.shape(x)[-1]
    y = jnp.stack([f(x[..., i], x[..., i+1], p[..., 0], p[..., 1]) for i in range(N-1)], axis=1)
    y = jnp.mean(y, axis=1)

    if single:
        y = jnp.squeeze(y, 0)
    
    return y 

def Sine(x):
    f = lambda x, y: 0.25*(jnp.sin(8*(x-1.0)-jnp.pi/2) + jnp.sin(8*(y-1.0)-jnp.pi/2)+2.0)

    single = x.ndim == 1
    if single:
        x = jnp.expand_dims(x, 0)

    N = jnp.shape(x)[-1]
    y = jnp.stack([f(x[..., i], x[..., i+1]) for i in range(N-1)], axis=1)
    y = jnp.mean(y, axis=1)

    if single:
        y = jnp.squeeze(y, 0)
    
    return y

def Sampler(
        rng: random.PRNGKey, 
        batches: int, 
        data_dim: int,
        x_min: Union[float, jnp.ndarray] = -2.,
        x_max: Union[float, jnp.ndarray] = 2., 
):
    return random.uniform(rng, (batches, data_dim), minval=x_min, maxval=x_max) 

def MeshField(
    x_range: Sequence[float] = [-2., 2.],
    y_range: Sequence[float] = [-1., 3.],
    n_grid: int = 200
):
    x = jnp.linspace(x_range[0], x_range[1], n_grid)
    y = jnp.linspace(y_range[0], y_range[1], n_grid)
    xx, yy = jnp.meshgrid(x, y)
    z = jnp.concatenate([jnp.reshape(xx,(-1,1)), jnp.reshape(yy, (-1,1))], axis=1)
    return z, xx, yy 

 