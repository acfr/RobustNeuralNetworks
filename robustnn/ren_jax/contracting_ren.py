import jax.numpy as jnp
from robustnn.ren_jax.ren_base import RENBase, DirectRENParams, ExplicitRENParams

class ContractingREN(RENBase):
    """Construct a Contracting REN.
    
    See docs for `RENBase` for full list of arguments.
    """
    
    def direct_to_explicit(self, ps: DirectRENParams) -> ExplicitRENParams:
        H = self.x_to_h(ps.X, ps.p)
        explicit = self.hmatrix_to_explicit(ps, H, ps.D22)
        return explicit