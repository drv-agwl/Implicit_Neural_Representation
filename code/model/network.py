import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad
from math import pi


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad


def doubleWellPotential(s):
    """
    double well potential function with zeros at -1 and 1
    """
    return (s ** 2) - 2 * (s.abs()) + 1.


class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, k):
        super().__init__()
        B = torch.randn(in_features, out_features) * k
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = torch.matmul(2 * pi * x, self.B)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out


class ImplicitNet(nn.Module):
    def __init__(
            self,
            FF,
            k,
            d_in,
            dims,
            skip_in=(),
            geometric_init=True,
            radius_init=1,
            beta=100
    ):
        super().__init__()

        self.FF = FF
        self.k = k

        if FF:
            self.ffLayer = FourierLayer(in_features=3, out_features=dims[0]//2, k=self.k)
            dims = [dims[0]] + dims + [1]
        else:
            dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, input):

        x = input

        if self.FF:
            x = self.ffLayer(x)  # apply the fourier

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
