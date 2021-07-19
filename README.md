# StyleGAN
An unofficial implementation of StyleGAN for educational purposes.

Currently, the StyleGAN config E architecture is implemented (without mixing regulation).

The following repositories were partially used when writing the architecture and train loop:
* [Style-Based GAN in PyTorch](https://github.com/rosinality/style-based-gan-pytorch) 
* [StyleGAN PyTorch](https://github.com/tomguluson92/StyleGAN_PyTorch)

As a loss function, WGAN-GP was taken from this implementation:
* [WGAN-GP](https://github.com/eriklindernoren/PyTorch-GAN/blob/a163b82beff3d01688d8315a3fd39080400e7c01/implementations/wgan_gp/wgan_gp.py)

It is recommended to run PyTorch version 1.9.0+cu102 or higher.
