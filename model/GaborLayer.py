import math
from typing import Any
import torch
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module
import cplxmodule.cplx

class GaborConv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
    ):
        super().__init__()

        self.is_calculated = False

        self.conv_layer_re = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.conv_layer_im = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer_re.kernel_size

        # small addition to avoid division by zero.
        self.delta = 1e-3

        # freq, theta, sigma are set up for the Gabor filter banks.
        self.freq = Parameter(
            (math.pi / 2)
            * math.sqrt(2)
            ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True,
        )
        self.theta = Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True,
        )
        self.sigma = Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = Parameter(
            math.pi * torch.rand(out_channels, in_channels), requires_grad=True
        )

        self.x0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False
        )
        self.y0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False
        )
        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ]
        )
        self.y = Parameter(self.y)
        self.x = Parameter(self.x)

        self.weight = Parameter(
            torch.empty(self.conv_layer_re.weight.shape, requires_grad=True),
            requires_grad=True,
        )

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        # return torch.cat((self.conv_layer_re(input_tensor), self.conv_layer_im(input_tensor)), dim=1)
        return torch.complex(self.conv_layer_re(input_tensor), self.conv_layer_im(input_tensor))

    def calculate_weights(self):
        for i in range(self.conv_layer_re.out_channels):
            for j in range(self.conv_layer_re.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(
                    -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
                )
                re = g * torch.cos(freq * rotx + psi)
                im = g * torch.sin(freq * rotx + psi)
                re = re / (2 * math.pi * sigma ** 2)
                im = im / (2 * math.pi * sigma ** 2)
                self.conv_layer_re.weight.data[i, j] = re
                self.conv_layer_im.weight.data[i, j] = im
