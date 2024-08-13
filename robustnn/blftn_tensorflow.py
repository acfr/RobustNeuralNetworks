import tensorflow as tf
from tensorflow.python import keras 
import numpy as np 

def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    
    _, m, n = W.shape 
    if n > m:
        W = tf.transpose(W, perm=[0,2,1])
        R = cayley(W)
        R = tf.transpose(W, perm=[0,2,1])
        return R
    
    U, V = W[:, :n, :], W[:, n:, :]
    Uc = tf.transpose(U, perm=[0,2,1], conjugate=True)
    Vc = tf.transpose(V, perm=[0,2,1], conjugate=True)
    Z = U - Uc + Vc @ V 
    I = tf.eye(n, dtype=Z.dtype)[None,:,:]
    Zi = tf.linalg.inv(I+Z) #tf.linalg.solve(I+Z, I)
    R = tf.concat([Zi @ (I-Z), -2 * V @ Zi], axis=1)

    return R

class MonLipNet(keras.layers.Layer):
    def __init__(self, input_dim, hidden_units, mu=0.1, nu=10.):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units 
        self.mu = mu 
        self.nu = nu 
        Fq = keras.initializers.GlorotUniform()((sum(hidden_units), input_dim))
        self.Fq = tf.Variable(Fq, name="Fq", trainable=True)
        fq = tf.reshape(tf.norm(Fq),(1,))
        self.fq = tf.Variable(fq, name="fq", trainable=True)
        by = keras.initializers.Zeros()((input_dim,))
        self.by = tf.Variable(by, name="by", trainable=True)
        nz_1 = 0
        Fr, fr, b = [], [], []
        for k, nz in enumerate(hidden_units):
            Rk = keras.initializers.GlorotUniform()((nz, nz+nz_1))
            Fr.append(tf.Variable(Rk, name=f"Fr{k}", trainable=True))
            rk = tf.norm(Rk)
            fr.append(tf.Variable(rk, name=f"fr{k}", trainable=True))
            bk = keras.initializers.Zeros()((nz,))
            b.append(tf.Variable(bk, name=f"b{k}", trainable=True))
            nz_1 = nz 
        self.Fr = Fr
        self.fr = fr
        self.b = b

    def call(self, x):
        sqrt_gam = tf.math.sqrt(self.nu - self.mu)
        sqrt_2 = tf.math.sqrt(2.)
        Q = cayley(self.fq / tf.norm(self.Fq) * self.Fq )
        xh = sqrt_gam * x @ tf.transpose(Q)
        yh = []
        idx = 0
        hk_1 = xh[..., :0]
        for k, nz in enumerate(self.hidden_units):
            xk = xh[..., idx:idx+nz]
            R = cayley(self.fr[k] / tf.norm(self.Fr[k]) * self.Fr[k])
            gh = tf.concat([xk, hk_1], axis=-1)
            gh = sqrt_2 * tf.nn.relu(sqrt_2 * gh @ tf.transpose(R) + self.b[k]) @ R
            hk = gh[..., :nz] - xk
            gk = gh[..., nz:]
            yh.append(hk_1-gk)
            idx += nz 
            hk_1 = hk 
        yh.append(hk_1)
        yh = tf.concat(yh, axis=-1)
        y = 0.5 * ((self.mu + self.nu) * x + sqrt_gam * yh @ Q) + self.by 
        return y  

class ConvMonLipNet(keras.layers.Layer):
    def __init__(self, cin, channels, kern_sz, img_sz, mu=0.1, nu=10.):
        super().__init__()
        self.cin = cin
        self.channels = channels 
        self.kern_sz = kern_sz
        self.img_sz = img_sz 
        self.mu = mu 
        self.nu = nu 
        self.fft_shift_matrix()
        Fq = keras.initializers.GlorotUniform()((sum(channels), cin, kern_sz, kern_sz))
        self.Fq = tf.Variable(Fq, name="Fq", trainable=True)
        Fqfft = self.fft_weight(Fq)
        fq = tf.cast(tf.norm(Fqfft), Fq.dtype)
        self.fq = tf.Variable(fq, name="fq", trainable=True)
        by = keras.initializers.Zeros()((cin,))
        self.by = tf.Variable(by, name="by", trainable=True)

        nz_1 = 0
        Fr, fr, b = [], [], []
        for k, nz in enumerate(channels):
            R = keras.initializers.GlorotUniform()((nz, nz+nz_1, kern_sz, kern_sz))
            Fr.append(tf.Variable(R, name=f"Fr{k}", trainable=True))
            Rfft = self.fft_weight(R)
            r = tf.cast(tf.norm(Rfft), dtype=R.dtype)
            fr.append(tf.Variable(r, name=f"fr{k}", trainable=True))
            bk = keras.initializers.Zeros()((nz,))
            b.append(tf.Variable(bk, name=f"b{k}", trainable=True))
            nz_1 = nz 

        self.Fr = Fr
        self.fr = fr
        self.b = b

    def fft_weight(self, W):
        n = self.img_sz 
        cout, cin, _, _ = W.shape 
        Wfft = tf.signal.rfft2d(W, fft_length=(n,n))
        Wfft = tf.reshape(Wfft, (cout, cin, n * (n // 2 + 1)))
        Wfft = tf.transpose(Wfft, perm=[2,0,1], conjugate=True)
        Wfft = self.shift_matrix * Wfft
        return Wfft

    def fft_vector(self, x):
        batches, cin, n, _ = x.shape
        xfft = tf.signal.rfft2d(x, fft_length=(n, n))
        xfft = tf.transpose(xfft, perm=[2, 3, 1, 0])
        xfft = tf.reshape(xfft, (n * (n // 2 + 1), cin, batches))
        return xfft 
    
    def ifft_vector(self, xfft):
        n = self.img_sz 
        _, channels, batches = xfft.shape 
        xfft = tf.reshape(xfft, (n, n//2+1, channels, batches))
        xfft = tf.transpose(xfft, perm=[3, 2, 0, 1])
        x = tf.signal.irfft2d(xfft)
        return x 

    def fft_shift_matrix(self):
        n = self.img_sz 
        s = -((self.kern_sz - 1) // 2)
        shift = np.reshape(np.arange(0, n), (1, n))
        shift = np.repeat(shift, repeats=n, axis=0)
        shift = shift + np.transpose(shift)
        shift = np.exp(1j * 2 * np.pi * s * shift / n)
        shift = tf.convert_to_tensor(shift, dtype=tf.complex64)
        shift = shift[:, :(n//2 + 1)]
        shift = tf.reshape(shift, (n * (n//2 + 1), 1, 1))
        self.shift_matrix = shift 

    def call(self, x):
        sqrt_gam = (self.nu - self.mu) ** 0.5
        sqrt_2 = (2.) ** 0.5
        Fqfft = self.fft_weight(self.Fq)
        Qfft = cayley(tf.cast(self.fq, Fqfft.dtype) / tf.norm(Fqfft) * Fqfft)
 
        xhfft = self.fft_vector(sqrt_gam * x)
        xhfft = Qfft @ xhfft
        yhfft = []
        hk_1fft = xhfft[:, :0, :]
        idx = 0
        for k, nz in enumerate(self.channels):
            xkfft = xhfft[:, idx:idx+nz, :]
            Frfft = self.fft_weight(self.Fr[k])
            Rfft = cayley(tf.cast(self.fr[k], Frfft.dtype) / tf.norm(Frfft) * Frfft)
            ghfft = tf.concat([xkfft, hk_1fft], axis=1)
            ghfft = Rfft @ ghfft 
            gh = self.ifft_vector(ghfft)
            gh = sqrt_2 * tf.nn.relu(sqrt_2 * gh + self.b[k][:, None, None])
            ghfft = self.fft_vector(gh)
            ghfft = tf.transpose(Rfft, perm=[0,2,1], conjugate=True) @ ghfft
            hkfft = ghfft[:, :nz, :] - xkfft
            gkfft = ghfft[:, nz:, :]
            yhfft.append(hk_1fft-gkfft)
            idx += nz 
            hk_1fft = hkfft 
        yhfft.append(hk_1fft)
        yhfft = tf.transpose(Qfft, perm=[0,2,1], conjugate=True) @ tf.concat(yhfft, axis=1)
        yh = self.ifft_vector(yhfft)
        y = 0.5 * ((self.mu + self.nu) * x + sqrt_gam * yh) + self.by[:, None, None] 
        return y

if __name__ == "__main__":
    b = 128
    n = 10
    eps = 0.05
    units = [16, 32, 64]
    model = MonLipNet(input_dim=n, hidden_units=units, mu=0.4, nu=2.0)
    x  = tf.random.normal((b,n))
    dx = tf.random.normal((b,n))
    y1 = model(x + eps*dx)
    y2 = model(x - eps*dx)
    dy = y1 - y2 
    ndx = tf.norm(dx, axis=1)
    ndy = tf.norm(dy, axis=1)
    lip = ndy / (2*eps*ndx)
    print(f"Lip max={tf.experimental.numpy.max(lip):.2f}, min={tf.experimental.numpy.min(lip):.2f}")

    img_sz = 16 
    kern_sz = 3 
    model = ConvMonLipNet(n, units, kern_sz, img_sz, mu=0.4, nu=2.0)
    x  = tf.random.normal((b, n, img_sz, img_sz))
    dx = tf.random.normal((b, n, img_sz, img_sz))
    y1 = model(x + eps*dx)
    y2 = model(x - eps*dx)
    dy = y1 - y2 
    dx = tf.reshape(dx, (b, n * img_sz * img_sz))
    dy = tf.reshape(dy, (b, n * img_sz * img_sz))
    ndx = tf.norm(dx, axis=1)
    ndy = tf.norm(dy, axis=1)
    lip = ndy / (2*eps*ndx)
    print(f"Lip max={tf.experimental.numpy.max(lip):.2f}, min={tf.experimental.numpy.min(lip):.2f}")
