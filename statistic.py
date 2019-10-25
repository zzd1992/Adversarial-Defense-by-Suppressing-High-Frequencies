import torch
import numpy as np


class SpectralEnergy(torch.nn.Module):
    def __init__(self, w, h):
        super(SpectralEnergy, self).__init__()
        self.w = w
        self.h = h
        self._make_interval()

    def _make_interval(self):
        if self.w % 2 == 0:
            self.w_s, self.w_r = 0, self.w // 2
        else:
            self.w_s, self.w_r = 1, self.w // 2

        if self.h % 2 == 0:
            self.h_s, self.h_r = 0, self.h // 2
        else:
            self.h_s, self.h_r = 1, self.h // 2

    def forward(self, x):
        y = torch.rfft(x, 2, onesided=False, normalized=True)
        y = y[..., 0] ** 2 + y[..., 1] ** 2
        y[:, :, self.h_s:self.h_s+self.h_r] += y[:, :, -self.h_r:][:, :, torch.arange(self.h_r-1, -1, -1)]
        y[..., self.w_s:self.w_s+self.w_r] += y[..., -self.w_r:][..., torch.arange(self.w_r-1, -1, -1)]

        return y[:, :, :self.h_s+self.h_r, :self.w_s+self.w_r]


def spectral_energy(x):
    '''
    :param x: numpy array
    :return:
    '''
    y = np.cumsum(x, 0)
    y = np.cumsum(y, 1)
    y = np.diag(y)

    return y / y[-1]


if __name__=='__main__':
    N = 25
    batchsize = 4
    F = SpectralEnergy(N, N)
    x = torch.rand(batchsize, 3, N, N)
    y = F(x)
    y = y.mean(0).mean(0)
    y = spectral_energy(y.numpy())
    print(y)
