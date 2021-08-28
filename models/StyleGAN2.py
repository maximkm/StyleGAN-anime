from collections import OrderedDict
from utils import register
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


generators = register.ClassRegistry()
discriminators = register.ClassRegistry()


def Normalize_2D(x, eps=1e-8):
    # Normalization std
    return x * (x.square().mean(-1) + eps).rsqrt().view(-1, 1)


def PixelNorm(img, eps=1e-8):
    # Normalization mean and std of the channels of the 4D tensor
    assert len(img.shape) == 4
    img = img - torch.mean(img, (2, 3), True)
    tmp = torch.mul(img, img)  # or x ** 2
    tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + eps)
    return img * tmp


def Generate_map_channels(out_res, start_res=4, max_channels=512):
    # Returns the number of channels for each intermediate resolution
    base_channels = 16 * 1024
    map_channels = dict()
    k = start_res
    while k <= out_res:
        map_channels[k] = min(base_channels // k, max_channels)
        k *= 2
    return map_channels


class Mapping(nn.Module):
    def __init__(self,
                 z_dim: int,           # Dimension of the latent space
                 deep_mapping=8,       # Mapping depth
                 normalize=True,       # Normalization of input vectors
                 eps=1e-8,             # Parameter for normalization stability
                 ):
        super().__init__()
        self.dim = z_dim
        self.deep = deep_mapping
        self.normalize = normalize
        self.eps = eps
        self.blocks = []

        # Creating blocks
        for i in range(self.deep):
            self.blocks.append(nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.LeakyReLU(0.2),
            ))

            # Initializing weights
            nn.init.xavier_normal_(self.blocks[-1][0].weight.data)
            nn.init.zeros_(self.blocks[-1][0].bias.data)

        # Registering parameters in the model
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, z):
        if self.normalize:
            z = Normalize_2D(z, self.eps)
        for block in self.blocks:
            z = block(z)
        return z

class Conv2Demod(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 demod=True,
                 eps=1e-8
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.demod = demod
        self.eps = eps
        
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size, kernel_size)))
        nn.init.xavier_normal_(self.weight)


    def __get_same_padding(self, size):
        # reference: https://stats.stackexchange.com/a/410270
        return ((self.stride - 1)*size - self.stride + self.kernel_size) // 2


    def forward(self, img, s):
        # reference: https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py
        b, c, height, weight = img.shape
        
        w1 = s.view(b, 1, -1, 1, 1)
        w2 = self.weight.view(1, *self.weight.shape)
        weights = w2 * (w1 + 1)
        
        if self.demod:
            d = ((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps).rsqrt()
            weights = weights * d
        
        img = img.reshape(1, b * c, height, weight)

        _, _, *ws = weights.shape
        weights = weights.view(b * self.out_channels, *ws)

        assert height == weight
        padding = self.__get_same_padding(height)
        img = F.conv2d(img, weights, padding=padding, groups=b)

        img = img.reshape(b, self.out_channels, height, weight)
        return img


class BlockG(nn.Module):
    def __init__(self,
                 res_in: int,          # Input image resolution
                 res_out: int,         # Output image resolution
                 in_channels: int,     # Input number of channels
                 out_channels: int,    # Output number of channels
                 latent_size: int,     # Dimension of the latent space
                 ):
        super().__init__()
        assert res_out == res_in or res_out == 2 * res_in
        self.res_in = res_in
        self.res_out = res_out
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size

        # Upsampling
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Creating layers
        self.Conv1 = Conv2Demod(in_channels, out_channels, 3, 1)
        self.Conv2 = Conv2Demod(out_channels, out_channels, 3, 1)
        self.A1 = nn.Linear(latent_size, in_channels)
        self.A2 = nn.Linear(latent_size, out_channels)
        self.Act = nn.LeakyReLU(0.2)

        # Weights for noise
        self.B1 = nn.Parameter(torch.zeros(out_channels))
        self.B2 = nn.Parameter(torch.zeros(out_channels))

        # Initializing weights
        nn.init.xavier_normal_(self.A1.weight.data)
        nn.init.zeros_(self.A1.bias.data)
        nn.init.xavier_normal_(self.A2.weight.data)
        nn.init.zeros_(self.A2.bias.data)

    def forward(self, x, w):
        assert len(x.shape) == 4

        # upsampling
        if self.res_out == 2 * self.res_in:
            x = self.up_sample(x)

        style1 = self.A1(w)
        x = self.Conv1(x, style1)
        x = self.Act(x)
        noise1 = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x = x + self.B1.view(1, -1, 1, 1) * noise1
        
        style2 = self.A2(w)
        x = self.Conv2(x, style2)
        x = self.Act(x)
        noise2 = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x = x + self.B2.view(1, -1, 1, 1) * noise2
        return x


@generators.add_to_registry("StyleGAN2")
class Generator(nn.Module):
    def __init__(self,
                 res: int,             # Generated image resolution
                 RGB=True,             # Color or gray image
                 deep_mapping=8,       # Mapping depth
                 start_res=4,          # Initial resolution of the constant
                 max_channels=512,     # Maximum number of channels in intermediate images
                 latent_size=512,      # Dimension of the latent space
                 normalize=True,       # Normalization of the input vector z
                 eps=1e-8,             # Parameter for normalization stability
                 ):
        super().__init__()
        assert 2 ** round(np.log2(res)) == res and res >= 4 and res <= 1024
        self.res = res
        self.out_channels = 3 if RGB else 1
        self.deep_mapping = deep_mapping
        self.latent_size = latent_size
        self.eps = eps

        # Calculating the number of channels for each resolution
        self.map_channels = Generate_map_channels(res, start_res, max_channels)

        # Initializing layers
        self.mapping = Mapping(latent_size, deep_mapping, normalize, eps)
        self.const = nn.Parameter(torch.ones(max_channels, start_res, start_res))
        self.blocks = OrderedDict([
            (f'res {start_res}', BlockG(start_res, start_res, max_channels, self.map_channels[start_res], latent_size)),
        ])
        self.to_rgb = nn.Conv2d(self.map_channels[res], self.out_channels, 1, 1)

        # Initializing weights
        nn.init.xavier_normal_(self.to_rgb.weight.data)
        nn.init.zeros_(self.to_rgb.bias.data)

        # Creating blocks
        to_res = 2 * start_res
        while to_res <= res:
            cur_res = to_res // 2
            in_channels = self.map_channels[cur_res]
            out_channels = self.map_channels[to_res]
            self.blocks[f'res {to_res}'] = BlockG(cur_res, to_res, in_channels, out_channels, latent_size)
            to_res *= 2

        # Registering parameters in the model
        self.blocks = nn.ModuleDict(self.blocks)

    def forward(self, z):
        w = self.mapping(z)
        img = self.const.expand(w.size(0), -1, -1, -1)
        for block in self.blocks.values():
            img = block(img, w)
        return self.to_rgb(img)


class BlockD(nn.Module):
    def __init__(self,
                 res_in: int,          # Input image resolution
                 res_out: int,         # Output image resolution
                 in_channels: int,     # Input number of channels
                 out_channels: int,    # Output number of channels
                 ):
        super().__init__()
        self.res_in = res_in
        self.res_out = res_out
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initializing layers
        self.Conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.Act = nn.LeakyReLU(0.2)
        self.Conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.down_sample = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        
        # initializing weights
        nn.init.xavier_normal_(self.Conv1.weight.data)
        nn.init.zeros_(self.Conv1.bias.data)
        nn.init.xavier_normal_(self.Conv2.weight.data)
        nn.init.zeros_(self.Conv2.bias.data)
        nn.init.xavier_normal_(self.down_sample.weight.data)
        nn.init.zeros_(self.down_sample.bias.data)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Act(x)
        x = self.Conv2(x)
        x = self.Act(x)
        x = self.down_sample(x)
        x = self.Act(x)
        return x


@discriminators.add_to_registry("StyleGAN2")
class Discriminator(nn.Module):
    def __init__(self,
                 res,                  # Input resolution of images
                 RGB=True,             # Color or gray images
                 last_res=4,           # The last resolution of the image before the linear layers
                 max_channels=512,     # Maximum number of channels in intermediate images
                 ):
        super().__init__()
        assert 2 ** round(np.log2(res)) == res
        self.res = res
        self.in_channels = 3 if RGB else 1
        self.blocks = OrderedDict()

        # Calculating the number of channels for each resolution
        self.map_channels = Generate_map_channels(res, last_res, max_channels)

        # Creating blocks
        to_res = res // 2
        while to_res >= last_res:
            cur_res = 2 * to_res
            in_channels = self.map_channels[cur_res]
            out_channels = self.map_channels[to_res]
            self.blocks[f'res {cur_res}'] = BlockD(cur_res, to_res, in_channels, out_channels)
            to_res //= 2

        self.fromRGB = nn.Conv2d(self.in_channels, self.map_channels[res], 1, 1)
        self.Linear = nn.Linear(self.map_channels[last_res] * last_res ** 2, 1)

        # initializing weights
        nn.init.xavier_normal_(self.Linear.weight.data)
        nn.init.zeros_(self.Linear.bias.data)

        # Registering parameters in the model
        self.blocks = nn.ModuleDict(self.blocks)

    def forward(self, img):
        assert img.shape[1: ] == (self.in_channels, self.res, self.res)
        
        img = img.view(-1, self.in_channels, self.res, self.res)
        img = self.fromRGB(img)
        for block in self.blocks.values():
            img = block(img)
        return self.Linear(img.view(img.size(0), -1))
