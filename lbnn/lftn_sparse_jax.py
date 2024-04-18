import jax 
import jax.numpy as jnp

from flax import linen as nn
import flax.linen.initializers as init
from typing import Any, Sequence, Callable


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def l2_norm(x, eps=jnp.finfo(jnp.float32).eps):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.maximum(jnp.sum(x**2), eps))


def cayley(W):
    """Perform Cayley transform on a stacked matrix [U; V]"""
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    U, V = W[:n, :], W[n:, :]
    Z = (U - U.T) + (V.T @ V)
    I = jnp.eye(n)
    ZI = Z + I
    
    # Note that B * A^-1 = solve(A.T, B.T).T
    return jnp.concatenate([jnp.linalg.solve(ZI, I-Z),
                            -2 * jnp.linalg.solve(ZI.T, V.T).T])


class LFTN_Sparse(nn.Module):
    """Lipschitz-bounded Feed-through Network.
    
    Example usage::
    
        >>> from liprl.networks.lftn import LFTN
        >>> import jax, jax.numpy as jnp
        
        >>> nu, ny = 5, 2
        >>> layers = (8, 16, ny)
        >>> gamma = jnp.float32(10)
        
        >>> model = LFTN(layer_sizes=layers, gamma=gamma)
        >>> params = model.init(jax.random.key(0), jnp.ones((6,nu)))
        >>> jax.tree_map(jnp.shape, params)
        {'params': {'Fq': (7, 24), 'Fr0': (8, 8), 'Fr1': (24, 16), 'b0': (8,), 'b1': (16,), 'by': (2,), 'fq': (1,), 'fr0': (1,), 'fr1': (1,), 'gamma': (1,)}}
    
    Attributes:
        layer_sizes: Tuple of hidden layer sizes and the output size.
        gamma: Upper bound on the Lipschitz constant (default: 1.0).
        activation: Activation function to use (default: relu).
        kernel_init: Initialisation function for matrics (default: glorot_normal).
        activate_final: Whether to apply activation to the final layer (default: False).
        use_bias: Whether to use bias terms (default: True).
        trainable_lipschitz: Whether to make the Lipschitz constant trainable (default: False).
    
    TODO: Optional activation on final layer is not implemented yet.
    """
    layer_sizes: Sequence[int]
    skip_connections: Sequence[bool]
    gamma: jnp.float32 = 1.0
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = init.glorot_normal()
    activate_final: bool = False
    use_bias: bool = True
    trainable_lipschitz: bool = False
    
    def setup(self):
        """Define some common sizes."""
        self.hidden_sizes = self.layer_sizes[:-1]
        self.output_size = self.layer_sizes[-1]

    
    @nn.compact
    def __call__(self, x : jnp.array) -> jnp.array:      
        
        # Input and output shapes
        nx = jnp.shape(x)[-1]
        ny = self.output_size
        
        # Set up trainable/constant Lipschitz bound
        gamma = self.param("gamma", init.constant(self.gamma), (1,), jnp.float32)
        if not self.trainable_lipschitz:
            _rng = jax.random.PRNGKey(0)
            gamma = init.constant(self.gamma)(_rng, (1,), jnp.float32)
        
        # Define free parameter Fq and compute Qx, Qy via normalized Cayley
        Fqx = self.param("Fq", self.kernel_init, (nx+ny, sum(self.hidden_sizes*self.skip_connections[:-1])), jnp.float32)
        fqx = self.param("fq", init.constant(l2_norm(Fqx)),(1,), jnp.float32)
        
        Qx = cayley((fqx / l2_norm(Fqx)) * Fqx)
        Fqy = self.param('Fqy', self.kernel_init, (nz, ny), jnp.float32)
        Qy = Fqy / l2_norm(Fqy)


        # Set up for layer-wise loop
        # xhat = jnp.sqrt(2*gamma) * x @ QT_x
        hk_1 = x[..., :0]
        yhat_ks = []
        idx = 0
        nz_1 = 0
        
        # Loop through the hidden layers
        for k, nz in enumerate(self.hidden_sizes):
            
            # Wrapper for scaling activation function
            if self.activation == nn.relu:
                activation_k = self.activation
            else:
                d = self.param(f'd{k}', init.zeros_init(), (nz,), jnp.float32)
                psi_k = jnp.exp(d)
                activation_k = lambda x: psi_k * self.activation(x / psi_k)
            
            # Free params Fr = [Fa; Fb] and get Rk = [Ak Bk]
            Fr = self.param(f"Fr{k}", self.kernel_init, (nz+nz_1, nz), jnp.float32)
            fr = self.param(f"fr{k}", init.constant(l2_norm(Fr)), (1,), jnp.float32)
            RT = cayley((fr / l2_norm(Fr)) * Fr)

            # Compute the layer update
            if self.skip_connections[k] == True:
                xhat_hk = jnp.sqrt(2*self.gamma1) * x @ Qx[:, k*nz:(k+1)*nz]
            else:
                xhat_hk = x @ jnp.zeros((nx, nz), jnp.float32)

            xhat_hk_1 = jnp.concatenate((xhat_hk, hk_1), axis=-1)
            temp_var = jnp.sqrt(2) * xhat_hk_1 @ RT
            
            if self.use_bias:
                bk = self.param(f'b{k}', init.zeros_init(), (nz,), jnp.float32)
                temp_var += bk
                
            gk_hk = jnp.sqrt(2) * activation_k(temp_var) @ RT.T
            
            # Split outputs and store for later
            hk = gk_hk[..., :nz] - xhat_hk
            gk = gk_hk[..., nz:]
            yhat_ks.append(hk_1 - gk)
            
            # Update intermediates/indices
            idx += nz
            hk_1 = hk
            nz_1 = nz
            
        # Handle the output layer separately
        # TODO(Ray) these differ a bit from the densely connected to the sparsely connect? Can you check please?
        yhat_ks.append(hk_1)
        yhat = jnp.concatenate(yhat_ks, axis=-1)
        y = jnp.sqrt(gamma/2) * (yhat) @ Qy.T
        
        if self.use_bias:
            by = self.param("by", init.zeros_init(), (ny,), jnp.float32)
            y += by
        
        return y