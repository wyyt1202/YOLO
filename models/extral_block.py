import math
import numpy as np
import torch.nn as nn
import torch
from models.attention import *
from einops import rearrange

class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super().__init__()
        self.c = c2 // 2  # Half of the output channels
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)  # First convolution layer: input channels = c1 // 2, output channels = c2 // 2, kernel size = 3, stride = 2, padding = 1
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)  # Second convolution layer: input channels = c1 // 2, output channels = c2 // 2, kernel size = 1, stride = 1, padding = 0

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)  # Average pooling with kernel size = 2, stride = 1, padding = 0
        x1, x2 = x.chunk(2, 1)  # Split input x into two parts along the channel dimension
        x1 = self.cv1(x1)  # Apply convolution to the first part
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)  # Max pooling on the second part with kernel size = 3, stride = 2, padding = 1
        x2 = self.cv2(x2)  # Apply convolution to the second part
        return torch.cat((x1, x2), 1)  # Concatenate the two parts along the channel dimension

class MCAM(nn.Module):
    def __init__(self, inc, kernel_sizes, e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)  # Hidden channels calculated based on input channels and ratio e
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # Upsampling layer with scale factor = 2
            Conv(inc[0], hidc, 1)  # Convolution layer: input channels = inc[0], output channels = hidc, kernel size = 1
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()  # If e != 1, use convolution; otherwise, use identity mapping
        self.conv3 = ADown(inc[2], hidc)  # Process the third input using the ADown module
               
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)  # List of depthwise separable convolutions: input channels = hidc, output channels = hidc * 3, kernel size = k, padding = autopad(k), groups = hidc * 3
        self.pw_conv = Conv(hidc * 3, hidc * 3)  # Pointwise convolution: input channels = hidc * 3, output channels = hidc * 3
    
    def forward(self, x):
        x1, x2, x3 = x  # Unpack the input x
        x1 = self.conv1(x1)  # Process x1
        x2 = self.conv2(x2)  # Process x2
        x3 = self.conv3(x3)  # Process x3
        x = torch.cat([x1, x2, x3], dim=1)  # Concatenate processed x1, x2, x3 along the channel dimension
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)  # Sum the outputs of x and each depthwise separable convolution
        feature = self.pw_conv(feature)  # Apply pointwise convolution to the summed feature
        x = x + feature  # Add the original x and the processed feature
        return x
        
class LS_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)  # Depthwise convolution: input channels = dim, output channels = dim, kernel size = 7, groups = dim, no activation
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)  # First 1x1 convolution: input channels = dim, output channels = mlp_ratio * dim
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)  # Second 1x1 convolution: input channels = dim, output channels = mlp_ratio * dim
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)  # 1x1 convolution: input channels = mlp_ratio * dim, output channels = dim, no activation
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)  # Second depthwise convolution: input channels = dim, output channels = dim, kernel size = 7, stride = 1, padding = (7-1)//2, groups = dim
        self.act = nn.ReLU6()  # ReLU6 activation
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath layer: use DropPath if drop_path > 0, else use identity mapping

    def forward(self, x):
        input = x  # Save the input x
        x = self.dwconv(x)  # Apply depthwise convolution to x
        x1, x2 = self.f1(x), self.f2(x)  # Apply two 1x1 convolutions to x
        x = self.act(x1) * x2  # Multiply the ReLU6-activated x1 with x2
        x = self.dwconv2(self.g(x))  # Apply 1x1 convolution followed by depthwise convolution
        x = input + self.drop_path(x)  # Add the original input and the processed x
        return x

class C3_LS(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # Hidden channels calculated based on output channels and ratio e
        self.m = nn.Sequential(*(LS_Block(c_) for _ in range(n)))  # Sequence of n LS_Block modules
