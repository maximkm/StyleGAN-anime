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
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(Images_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Create the model
    start_epoch = 0
    G = Generator(IMG_SIZE, latent_size=LATENT)
    D = Discriminator(IMG_SIZE)
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
    optim_D = torch.optim.Adam(D.parameters(), lr=0.003, betas=(0.45, 0.5))
    optim_G = torch.optim.Adam(G.parameters(), lr=0.003, betas=(0.45, 0.5))
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)

    def r1loss(inputs, label=None):
        # non-saturating loss with R1 regularization
        l = -1 if label else 1
        return F.softplus(l * inputs).mean()

    # Train
    fix_z = torch.randn([BATCH_SIZE, LATENT]).to(device)
    for epoch in range(start_epoch, epochs):
        bar = tqdm(dataloader)
        loss_D_list = []
        loss_G_list = []
        for i, (real_img, _) in enumerate(bar):
            # (1) Update D network
            D.zero_grad()

            real_img = real_img.to(device)
            real_img.requires_grad = True
            real_logit = D(real_img)
            d_real_loss = r1loss(real_logit, True)

            grad_real = grad(outputs=real_logit.sum(), inputs=real_img, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = 0.5 * R1_GAMMA * grad_penalty
            D_x_loss = d_real_loss + grad_penalty

            fake_img = G(torch.randn([real_img.size(0), LATENT]).to(device))
            fake_logit = D(fake_img.detach())
            D_z_loss = r1loss(fake_logit, False)
            D_loss = D_x_loss + D_z_loss

            D.zero_grad()
            D_loss.backward()
            optim_D.step()

            loss_D_list.append(D_loss.item())

            # (2) Update G network
            if i % UPD_FOR_GEN == 0:
                G.zero_grad()
                fake_img = G(torch.randn([real_img.size(0), LATENT]).to(device))
                fake_logit = D(fake_img)
                G_loss = r1loss(fake_logit, True)

                G_loss.backward()
                optim_G.step()

                loss_G_list.append(G_loss.item())

            # Generate test image
            if (i + 1) % (len(dataloader) // 4) == 0:
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
                    plt.imshow(G(torch.randn(LATENT).to(device)).detach().cpu()[0].permute(1, 2, 0))
                    plt.show()
            bar.set_description(f"Epoch {epoch + 1}/{epochs} [{i + 1}, {len(dataloader)}] [G]: {loss_G_list[-1]} [D]: {loss_D_list[-1]}")

        # Save the result
        Loss_G_list.append(np.mean(loss_G_list))
        Loss_D_list.append(np.mean(loss_D_list))

        # Save model
        state = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'Loss_G': Loss_G_list,
            'Loss_D': Loss_D_list,
            'start_epoch': epoch,
        }
        torch.save(state, f'{Weight_dir}/weight.pth')

        scheduler_D.step()
        scheduler_G.step()

    # Plot all loss train
    clear_output(wait=True)
    plt.figure()
    plt.plot(Loss_D_list, '-o')
    plt.title("Loss curve (Discriminator)")
    plt.show()

    plt.figure()
    plt.plot(Loss_G_list, '-o')
    plt.title("Loss curve (Generator)")
    plt.show()


if __name__ == "__main__":
    # Setting up the main training parameters
    IMG_SIZE = 64
    UPD_FOR_GEN = 1
    BATCH_SIZE = 32
    LATENT = 512
    R1_GAMMA = 10
    Images_dir = 'CelebA'
    Weight_dir = 'WeightGAN'
    output = 'ResultGAN'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    train()
