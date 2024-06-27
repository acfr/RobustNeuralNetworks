import jax.numpy as jnp
from robustnn.ren_jax.ren_base import RENBase

class ContractingREN(RENBase):
    """Construct a Contracting REN.
    
    See docs for `RENBase` for full list of arguments.
    """
    
    def direct_to_explicit(self, 
                           X, p, B2, D12, Y1, C2, D21, 
                           D22, X3, Y3, Z3, bx, bv, by):
        H = self.x_to_h(X, p)
        explicit = self.hmatrix_to_explicit(H, B2, D12, Y1, C2, 
                                            D21, D22, bx, bv, by)
        return explicit