import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

def cayley(W: torch.Tensor) -> torch.Tensor:
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)

    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)
        
class Conv2dFTN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 unit_channels: Sequence[int],
                 out_channels: int,
                 kernel_size: int,
                 image_size: int,
                 gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma 
        self.units = unit_channels
        self.cin = in_channels
        self.cout = out_channels
        self.kernel_size = kernel_size
        self.image_size = image_size 
        self.Fq = nn.Parameter(torch.empty(sum(self.units), in_channels + out_channels,  kernel_size, kernel_size))
        nn.init.xavier_normal_(self.Fq)
        self.fq = nn.Parameter(torch.zeros((1,)))
        self.by = nn.Parameter(torch.zeros(out_channels))

        Fr, fr, b = [], [], []
        nz_1 = 0
        for nz in self.units:
            R = nn.Parameter(torch.empty((nz, nz+nz_1, kernel_size, kernel_size)))
            nn.init.xavier_normal_(R)
            Fr.append(R)
            fr.append(nn.Parameter(torch.zeros((1,))))
            b.append(nn.Parameter(torch.zeros(nz)))
            nz_1 = nz
        self.Fr = nn.ParameterList(Fr)
        self.fr = nn.ParameterList(fr)
        self.b = nn.ParameterList(b)
        self.reset_params()
        
        # cached weights
        self.Qfft = None 
        self.Rfft = None  

    def reset_params(self):
        n = self.image_size
        s = (self.kernel_size - 1) // 2
        shift_matrix = self.fft_shift_matrix(
                n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1)
        cout, cin, _, _ = self.Fq.shape 
        Fqfft = shift_matrix * torch.fft.rfft2(self.Fq, (n, n)).reshape(
            cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        self.fq.data = Fqfft.norm()
        for k, Fr in enumerate(self.Fr):
            cout, cin, _, _ = Fr.shape 
            Frfft = shift_matrix * torch.fft.rfft2(Fr, (n, n)).reshape(
                cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            self.fr[k].data = Frfft.norm()

    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * torch.pi * s * shift / n)
    
    def get_weight_fft(self):
        n = self.image_size
        cout, cin, _, _ = self.Fq.shape 
        Fqfft = self.shift_matrix * torch.fft.rfft2(self.Fq, (n, n)).reshape(
            cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        Qfft = cayley(self.fq * Fqfft / Fqfft.norm())
        Rfft = []
        for Fr, fr in zip(self.Fr, self.fr):
            cout, cin, _, _ = Fr.shape 
            Frfft = self.shift_matrix * torch.fft.rfft2(Fr, (n, n)).reshape(
                cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            Rfft.append(cayley(fr * Frfft / Frfft.norm()))
        return Qfft, Rfft 
    
    def forward(self, x):
        batches, nx, n, _ = x.shape

        if not hasattr(self, 'shift_matrix'):
            s = (self.kernel_size - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(
                n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
            
        if self.training:
            self.Qfft, self.Rfft = None, None
            Qfft, Rfft = self.get_weight_fft()
        else:
            if self.Qfft is None:
                self.Qfft, self.Rfft = self.get_weight_fft()
            Qfft, Rfft = self.Qfft, self.Rfft

        x = (self.gamma ** 0.5) * x 
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(
            n * (n // 2 + 1), nx, batches)
        xhfft = Qfft[..., :nx] @ xfft 
        yhfft = []
        hk_1fft = xhfft[:, :0, :]
        idx = 0 
        for k, nz in enumerate(self.units):
            xkfft = xhfft[:, idx:idx+nz, :]
            zfft = (Rfft[k] @ torch.cat((xkfft, hk_1fft), dim=1)).reshape(n, n // 2 + 1, nz, batches)
            z = torch.fft.irfft2(zfft.permute(3, 2, 0, 1))
            z = (2 ** 0.5) * F.relu ((2 ** 0.5) * z  + self.b[k][:, None, None]) 
            zfft = torch.fft.rfft2(z).permute(2, 3, 1, 0).reshape(
                n * (n // 2 + 1), nz, batches)
            ghfft = Rfft[k].conj().transpose(1,2) @ zfft 
            hkfft = ghfft[:, :nz, :] - xkfft 
            gkfft = ghfft[:, nz:, :]
            yhfft.append(hk_1fft-gkfft)
            idx += nz 
            hk_1fft = hkfft
        yhfft.append(hk_1fft)
        yhfft = torch.cat(yhfft, dim=1)
        yfft = Qfft[..., nx:].conj().transpose(1,2) @ (xhfft + yhfft)
        yfft = (yfft).reshape(n, n // 2 + 1, self.cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        y = (self.gamma / 2) ** 0.5 * y + self.by[:, None, None]

        return y

class FTN(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 unit_features: Sequence[int],
                 out_features: int,
                 gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma 
        self.units = unit_features
        self.Fq = nn.Parameter(torch.empty(sum(self.units), in_features+out_features))
        nn.init.xavier_normal_(self.Fq)
        self.fq = nn.Parameter(torch.empty((1,)))
        nn.init.constant_(self.fq, self.Fq.norm())
        self.by = nn.Parameter(torch.zeros(out_features))
        Fr, fr, b = [], [], []
        nz_1 = 0
        for nz in self.units:
            R = nn.Parameter(torch.empty((nz, nz+nz_1)))
            nn.init.xavier_normal_(R)
            r = nn.Parameter(torch.empty((1,)))
            nn.init.constant_(r, R.norm())
            Fr.append(R)
            fr.append(r)
            b.append(nn.Parameter(torch.zeros(nz)))
            nz_1 = nz
        self.Fr = nn.ParameterList(Fr)
        self.fr = nn.ParameterList(fr)
        self.b = nn.ParameterList(b)
        # cached weights
        self.Q = None 
        self.R = None 

    def forward(self, x):
        nx = x.shape[-1] 
        if self.training:
            self.Q, self.R = None, None 
            Q = cayley(self.fq * self.Fq / self.Fq.norm())
            R = [cayley(fr * Fr / Fr.norm()) for Fr, fr in zip(self.Fr, self.fr)]
        else:
            if self.Q is None:
                with torch.no_grad():
                    self.Q = cayley(self.fq * self.Fq / self.Fq.norm())
                    self.R = [cayley(fr * Fr / Fr.norm()) for Fr, fr in zip(self.Fr, self.fr)]
            Q, R = self.Q, self.R 

        xh = (2*self.gamma) ** 0.5 * x @ Q[:, :nx].T
        yh = []
        hk_1 = xh[..., :0]
        idx = 0 
        for k, nz in enumerate(self.units):
            xk = xh[..., idx:idx+nz]
            gh = (2 ** 0.5) * F.relu ((2 ** 0.5) * torch.cat((xk, hk_1), dim=-1) @ R[k].T + self.b[k]) @ R[k]
            hk = gh[..., :nz] - xk
            gk = gh[..., nz:]
            yh.append(hk_1-gk)
            idx += nz 
            hk_1 = hk 
        yh.append(hk_1)

        yh = torch.cat(yh, dim=-1)
        y = (self.gamma / 2) ** 0.5 * (xh + yh) @ Q[:, nx:] + self.by 

        return y
