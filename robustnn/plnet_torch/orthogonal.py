import torch
import torch.nn as nn
import torch.nn.functional as F

class Params(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v)  # not learnable
            else:
                setattr(self, k, v)

def cayley(W: torch.Tensor) -> torch.Tensor:
    cout, cin = W.shape
    if cin > cout:
        return cayley(W.T).T
    U, V = W[:cin, :], W[cin:, :]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)
    A = U - U.T + V.T @ V
    iIpA = torch.inverse(I + A)

    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=0)

def norm(x, eps=0.0):
    return x.norm() + eps

class CayleyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.empty(1).fill_(
            norm(self.weight).item()), requires_grad=True)

        self.Q_cached = None

    def reset_parameters(self):
        std = 1 / self.weight.shape[1] ** 0.5
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

        self.Q_cached = None

    def forward(self, X):
        if self.training:
            self.Q_cached = None
            Q = cayley(self.alpha * self.weight / norm(self.weight))
        else:
            if self.Q_cached is None:
                with torch.no_grad():
                    self.Q_cached = cayley(
                        self.alpha * self.weight / norm(self.weight))
            Q = self.Q_cached

        # print(f"Q: {Q} and X: {X}")
        return F.linear(X, Q, self.bias)
    
    def inverse(self, y):
        """
        Inverse of the Cayley linear layer.
        Args:
            y (torch.Tensor): Input tensor to be inverted.
        Returns:
            torch.Tensor: Inverted tensor.
        """
        # todo: this is not correct, need to implement the non-tensor form
        bias_np = self.bias.numpy(force=True)
        # print(f'bias_np: {bias_np}')
        Q = cayley(self.alpha * self.weight / norm(self.weight, eps=0.0))
        Q_np = Q.numpy(force=True)
    
        return  (y - bias_np) @ Q_np
