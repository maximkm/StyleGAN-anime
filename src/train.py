import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import grad
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch

from IPython.display import clear_output
from matplotlib import pyplot as plt
from time import gmtime, strftime
from tqdm import tqdm
import numpy as np
import os
from src.StyleGAN import Generator, Discriminator


def train():
    # Load train image
    transform = transforms.Compose(
        [
         transforms.Resize((conf.IMG_SIZE, conf.IMG_SIZE)),
         transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5)),
         ]
        )

    dataset = torchvision.datasets.ImageFolder(conf.Dataset, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=0)

    # Create the model
    start_epoch = 0
    G = Generator(conf.IMG_SIZE, deep_mapping=conf.deep_mapping, latent_size=conf.LATENT, bilinear=conf.bilinear, channel_base=conf.channel_base)
    D = Discriminator(conf.IMG_SIZE, bilinear=conf.bilinear, channel_base=conf.channel_base)
    Loss_G_list = []
    Loss_D_list = []

    # Load the pre-trained weight
    if os.path.exists(f'{Weight_dir}/weight.pth'):
        print('Load the pre-trained weight')
        state = torch.load(f'{Weight_dir}/weight.pth')
        G.load_state_dict(state['G'])
        D.load_state_dict(state['D'])
        start_epoch = state['start_epoch']
        Loss_G_list = state['Loss_G']
        Loss_D_list = state['Loss_D']

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f'Avalible {torch.cuda.device_count()} GPUs')
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
    G.to(device)
    D.to(device)

    # Create the criterion, optimizer and scheduler
    optim_D = torch.optim.Adam(D.parameters(), lr=conf.lr_D, betas=(conf.beta1_D, conf.beta2_D))
    optim_G = torch.optim.Adam(G.parameters(), lr=conf.lr_G, betas=(conf.beta1_G, conf.beta2_G))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    def compute_gradient_penalty(D, real_samples, fake_samples):
        # Calculates the gradient penalty loss for WGAN GP
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

# Train
fix_z = torch.randn([conf.BATCH_SIZE, conf.LATENT]).to(device)
for epoch in range(start_epoch, conf.epochs):
    bar = tqdm(dataloader)
    loss_D_list = []
    loss_G_list = []
    for i, (real_img, _) in enumerate(bar):
        # (1) Update D network
        real_imgs = Variable(real_img.type(Tensor))
        optim_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (real_imgs.size(0), conf.LATENT))))

        # Generate a batch of images

        fake_imgs = G(z)

        # Real images
        real_validity = D(real_imgs)
        # Fake images
        fake_validity = D(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + conf.lambda_gp * gradient_penalty

        d_loss.backward()
        optim_D.step()
        
        loss_D_list.append(d_loss.item())
        
        optim_G.zero_grad()
        # (2) Update G network
        if i % conf.UPD_FOR_GEN == 0:
            # Generate a batch of images
            fake_imgs = G(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = D(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optim_G.step()
            
            loss_G_list.append(g_loss.item())
            wandb.log({"loss_G":g_loss.item(), "loss_D":d_loss.item()})

        # Generate test image
        if (i+1) % (len(dataloader) // 4) == 0:
            with torch.no_grad():
                fake_img = G(fix_z).detach().cpu()
                save_image(fake_img, f'{output}/{strftime("%Y-%m-%d %H-%M", gmtime())} {epoch}.png', normalize=True)

        # Output training stats
        if i % 10 == 0:
            clear_output(wait=True)
            plt.figure()
            plt.plot(loss_D_list[-20:-1], '-o')
            plt.title("Last 20 loss curve (Discriminator)")
            plt.show()

            plt.figure()
            plt.plot(loss_G_list[-20:-1], '-o')
            plt.title("Last 20 loss curve (Generator)")
            plt.show()

            with torch.no_grad():
                plt.title("Random generated face")
                plt.imshow(G(torch.randn(conf.LATENT).to(device)).detach().cpu()[0].permute(1, 2, 0))
                plt.show()
        bar.set_description(f"Epoch {epoch + 1}/{conf.epochs} [{i+1}, {len(dataloader)}] [G]: {loss_G_list[-1]} [D]: {loss_D_list[-1]}")

    # Save the result
    Loss_G_list.append(np.mean(loss_G_list))
    Loss_D_list.append(np.mean(loss_D_list))

    # Save model
    state = {
        'G': G.state_dict(),
        'D': D.state_dict(),
        'Loss_G': Loss_G_list,
        'Loss_D': Loss_D_list,
        'start_epoch': epoch + 1,
    }
    torch.save(state, f'{Weight_dir}/weight {epoch + 1}.pth')


if __name__ == "__main__":
    wandb.login()
    
    # Setting up the main training parameters
    wandb.init(
        project='StyleGAN-E',
        config={
            'IMG_SIZE': 32,
            'UPD_FOR_GEN': 1,
            'BATCH_SIZE': 16,
            'LATENT': 256,
            'lambda_gp': 10,
            'deep_mapping': 6,
            'channel_base': 16384,
            'bilinear': True,
            'lr_G': 0.003,
            'lr_D': 0.003,
            'beta1_G': 0.5,
            'beta2_G': 0.55,
            'beta1_D': 0.5,
            'beta2_D': 0.55,
            'Dataset': 'DataGAN1',
            'epochs': 100,
        }
    )
    conf = wandb.config
    Weight_dir = 'WeightGAN'
    output = 'ResultGAN2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
