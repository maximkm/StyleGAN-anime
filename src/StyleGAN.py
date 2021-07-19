import torch.nn as nn
import torch
import numpy as np


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
    base_channels = 16 * out_res
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
        for i in range(self.deep - 1):
            self.blocks.append(nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.LeakyReLU(0.2),
            ))

            # Initializing weights
            nn.init.xavier_normal_(self.blocks[-1][0].weight.data)
            nn.init.zeros_(self.blocks[-1][0].bias.data)

        self.blocks.append(nn.Linear(self.dim, self.dim))
        nn.init.xavier_normal_(self.blocks[-1].weight.data)
        nn.init.zeros_(self.blocks[-1].bias.data)

        # Registering parameters in the model
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, z):
        if self.normalize:
            z = Normalize_2D(z, self.eps)
        for block in self.blocks:
            z = block(z)
        return z


class AdaIN(nn.Module):
    def __init__(self,
                 latent_size: int,     # Dimension of the latent space
                 channels: int,        # The number of channels in the image
                 ):
        super().__init__()
        self.size = latent_size
        self.channels = channels

        # Affine transformation
        self.A = nn.Linear(self.size, 2 * channels)

        # Weights for noise
        self.B = nn.Parameter(torch.zeros(channels))

        # Initializing weights
        nn.init.xavier_normal_(self.A.weight.data)
        nn.init.zeros_(self.A.bias.data)

    def forward(self, x, w):
        # Apply noise
        noise = torch.randn(x.shape, device=x.device)
        x = x + self.B.view(1, -1, 1, 1) * noise

        # Apply style
        x = PixelNorm(x)
        style = self.A(w).view(2, -1, self.channels, 1, 1)
        x = (1 + style[0]) * x + style[1]
        return x


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
        self.AdaIN1 = AdaIN(self.latent_size, in_channels)
        self.Conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.Act = nn.LeakyReLU(0.2)
        self.AdaIN2 = AdaIN(self.latent_size, out_channels)

        # Initializing weights
        nn.init.kaiming_normal_(self.Conv.weight.data)
        nn.init.zeros_(self.Conv.bias.data)

    def forward(self, x, w):
        assert len(x.shape) == 4

        # upsampling
        if self.res_out == 2 * self.res_in:
            x = self.up_sample(x)

        x = self.AdaIN1(x, w)
        x = self.Conv(x)
        x = self.AdaIN2(x, w)
        return x


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
        assert 2 ** round(np.log2(res)) == res and res >= 4
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
        self.blocks = [
            BlockG(start_res, start_res, max_channels, self.map_channels[start_res], latent_size),
            BlockG(start_res, start_res, self.map_channels[start_res], self.map_channels[start_res], latent_size)
        ]
        self.to_rgb = nn.Conv2d(self.map_channels[res], self.out_channels, 1, 1)

        # Initializing weights
        nn.init.kaiming_normal_(self.to_rgb.weight.data)
        nn.init.zeros_(self.to_rgb.bias.data)

        # Creating blocks
        to_res = 2 * start_res
        while to_res <= res:
            cur_res = to_res // 2
            in_channels = self.map_channels[cur_res]
            out_channels = self.map_channels[to_res]
            self.blocks.append(BlockG(cur_res, to_res, in_channels, out_channels, latent_size))
            self.blocks.append(BlockG(to_res, to_res, out_channels, out_channels, latent_size))
            to_res *= 2

        # Registering parameters in the model
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, z):
        w = self.mapping(z)
        img = self.const.expand(w.size(0), -1, -1, -1)
        for block in self.blocks:
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
        self.Conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.act1 = nn.LeakyReLU(0.2)

        # initializing weights
        nn.init.kaiming_normal_(self.Conv1.weight.data)
        nn.init.zeros_(self.Conv1.bias.data)

        if 2 * res_out == res_in:
            self.down_sample = nn.AvgPool2d(3, 2, padding=1)
        else:
            self.Conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.act2 = nn.LeakyReLU(0.2)

            nn.init.kaiming_normal_(self.Conv2.weight.data)
            nn.init.zeros_(self.Conv2.bias.data)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.act1(x)
        if 2 * self.res_out == self.res_in:
            x = self.down_sample(x)
        else:
            x = self.Conv2(x)
            x = self.act2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,
                 res,                  # Input resolution of images
                 RGB=True,             # Color or gray images
                 last_res=1,           # The last resolution of the image before the linear layers
                 max_channels=512,     # Maximum number of channels in intermediate images
                 ):
        super().__init__()
        assert 2 ** round(np.log2(res)) == res
        self.res = res
        self.in_channels = 3 if RGB else 1
        self.blocks = []

        # Calculating the number of channels for each resolution
        self.map_channels = Generate_map_channels(res, last_res, max_channels)

        # Creating blocks
        to_res = res // 2
        while to_res >= last_res:
            cur_res = 2 * to_res
            in_channels = self.map_channels[cur_res]
            out_channels = self.map_channels[to_res]
            self.blocks.append(BlockD(cur_res, cur_res, in_channels, out_channels))
            self.blocks.append(BlockD(cur_res, to_res, out_channels, out_channels))
            to_res //= 2

        self.fromRGB = nn.Conv2d(self.in_channels, self.map_channels[res], 1, 1)
        self.Linear = nn.Linear(self.map_channels[last_res] * last_res ** 2, 1)

        # initializing weights
        nn.init.xavier_normal_(self.Linear.weight.data)
        nn.init.zeros_(self.Linear.bias.data)

        # Registering parameters in the model
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, img):
        img = img.view(-1, self.in_channels, self.res, self.res)
        img = self.fromRGB(img)
        for block in self.blocks:
            img = block(img)
        return self.Linear(img.view(img.size(0), -1))
