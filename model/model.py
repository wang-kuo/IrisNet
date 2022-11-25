from audioop import bias
from pyrsistent import s
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.jit.annotations import List
import torch.utils.checkpoint as cp
from collections import OrderedDict
import numpy as np
from .GaborLayer import GaborConv2d
import cplxmodule.cplx
from cplxmodule.nn.modules.batchnorm import CplxBatchNorm2d
from cplxmodule.nn.modules.activation import CplxModReLU
from cplxmodule.nn.modules.pooling import CplxMaxPool2d
from cplxmodule.nn.modules.conv import CplxConv2d
# class CplxModReLU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.mod.add_module('mod', CplxModReLU())
#     def forward(self, input: Tensor) -> Tensor:
#         return cplxmodule.cplx.Cplx(torch.relu(input.real), torch.relu(input.imag))

class GaborNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=20, kernel_size=(7, 7), stride=1, padding=3):
        super(GaborNN, self).__init__()
        self.gabor = GaborConv2d(in_channels, out_channels, kernel_size, stride, padding) 
        # self.add_module('gabor', CplxConv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)),
        self.add_module('norm', CplxBatchNorm2d(out_channels))
        self.add_module('relu', CplxModReLU())
        self.add_module('pool', CplxMaxPool2d(2))
    def forward(self, x):
        x = self.gabor(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', CplxBatchNorm2d(num_input_features)),
        self.add_module('relu1', CplxModReLU()),
        self.add_module('conv1', CplxConv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', CplxBatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', CplxModReLU()),
        self.add_module('conv2', CplxConv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = cplxmodule.cplx.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        # bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(torch.view_as_complex(new_features), p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return cplxmodule.cplx.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', CplxBatchNorm2d(num_input_features))
        self.add_module('relu', CplxModReLU())
        self.add_module('conv', CplxConv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', CplxMaxPool2d(2))


class ComplexIrisNet(nn.Sequential):
    
    def __init__(self):

        super(ComplexIrisNet, self).__init__()
        self.gabor = GaborNN(out_channels=20)
        self.dense1 = _DenseBlock(num_layers= 4, num_input_features=20, bn_size=4, growth_rate=16, drop_rate=0)
        self.tran1 = _Transition(num_input_features=84, num_output_features=16)
        self.dense2 = _DenseBlock(num_layers= 4, num_input_features=16, bn_size=4, growth_rate=16, drop_rate=0)
        self.tran2 = _Transition(num_input_features=80, num_output_features=10)
        # self.dense1 = _DenseBlock(num_layers= 6, num_input_features=20, bn_size=4, growth_rate=16, drop_rate=0)
        # self.tran1 = _Transition(num_input_features=84, num_output_features=16)
        # self.dense2 = _DenseBlock(num_layers= 6, num_input_features=32, bn_size=4, growth_rate=16, drop_rate=0)
        # self.tran2 = _Transition(num_input_features=80, num_output_features=10)

    def forward(self, x):
        out = self.gabor(x)
        out = self.dense1(out)
        out = self.tran1(out)
        out = self.dense2(out)
        out = self.tran2(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embeddingnet):
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        return torch.cat((embedded_x.real, embedded_x.imag),dim=1), torch.cat((embedded_y.real, embedded_y.imag),dim=1), torch.cat((embedded_z.real, embedded_z.imag),dim=1)
        # return embedded_x, embedded_y, embedded_z

class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()
        self.add_module('conv1', nn.Conv2d(1, 16, kernel_size=(3,7), stride=1, padding=(1,3), bias=False))
        self.add_module('tanh1', nn.Tanh())
        self.add_module('pool1', nn.AvgPool2d(kernel_size=(2,2), stride=2))
        self.add_module('conv2', nn.Conv2d(16, 32, kernel_size=(3,5), stride=1, padding=(1,2), bias=False))
        self.add_module('tanh2', nn.Tanh())
        self.add_module('pool2', nn.AvgPool2d(kernel_size=(2,2), stride=2))
        self.add_module('conv3', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('tanh3', nn.Tanh())
        self.add_module('up1', nn.Upsample(scale_factor=2, mode='bilinear'))
        self.add_module('up2', nn.Upsample(scale_factor=4, mode='bilinear'))
        self.add_module('conv4', nn.Conv2d(112, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('tanh4', nn.Tanh())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, input):
        s1 = self.tanh1(self.conv1(input))
        s2 = self.tanh2(self.conv2(self.pool1(s1)))
        s3 = self.tanh3(self.conv3(self.pool2(s2)))
        s4 = torch.cat((s1, self.up1(s2), self.up2(s3)), dim=1)
        s5 = self.tanh4(self.conv4(s4))
        return s5
    
class QuadletNet(nn.Module):
    def __init__(self, embeddingnet):
        super(QuadletNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z, z2):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        embedded_z2 = self.embeddingnet(z2)
        return embedded_x, embedded_y, embedded_z, embedded_z2