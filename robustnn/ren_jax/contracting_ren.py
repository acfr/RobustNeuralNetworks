import jax.numpy as jnp
from robustnn.ren_jax.ren_base import RENBase, DirectRENParams, ExplicitRENParams

class ContractingREN(RENBase):
    """Construct a Contracting REN.
    
    Example usage::

        >>> from robustnn.ren_jax.contracting_ren import ContractingREN
        >>> import jax, jax.numpy as jnp
        
        >>> rng = jax.random.key(0)
        >>> key1, key2 = jax.random.split(rng)

        >>> nu, nx, nv, ny = 1, 2, 4, 1
        >>> ren = ContractingREN(nu, nx, nv, ny)
        
        >>> batches = 5
        >>> states = ren.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = ren.init(key2, states, inputs)
        >>> jax.tree_map(jnp.shape, params)
        {'params': {'B2': (2, 1), 'C2': (1, 2), 'D12': (4, 1), 'D21': (1, 4), 'D22': (1, 1), 'X': (8, 8), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bv': (4,), 'bx': (2,), 'by': (1,), 'polar': (1,)}}
    
    See docs for `RENBase` for full list of arguments.
    """
    d22_free: bool = True
    
    def direct_to_explicit(self, ps: DirectRENParams) -> ExplicitRENParams:
        H = self.x_to_h(ps.X, ps.p)
        explicit = self.hmatrix_to_explicit(ps, H, ps.D22)
        return explicit