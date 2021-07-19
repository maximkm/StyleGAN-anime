from src.StyleGAN import Generator, Discriminator
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import grad, Variable
from torchvision.utils import save_image
import torch.nn as nn
import torchvision
import torch

from IPython.display import clear_output, HTML, display
from matplotlib import animation, pyplot as plt
from time import gmtime, strftime
from tqdm import tqdm
from PIL import Image
import numpy as np
import wandb
import os


class ImageDataset(Dataset):
    def __init__(self, data_root, transform):
        self.samples = []
        self.transform = transform

        for class_dir in os.listdir(data_root):
            data_folder = os.path.join(data_root, class_dir)

            for image_dir in tqdm(os.listdir(data_folder)):
                img = Image.open(f'{data_folder}/{image_dir}')
                img = img.convert("RGB")
                self.samples.append(self.transform(img))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def TensorToImage(img, mean=0.5, std=0.28):
    # Convert a tensor to an image
    img = np.transpose(img.numpy(), (1, 2, 0))
    img = (img*std + mean)*255
    img = img.astype(np.uint8)
    return img


def train():
    # Load train image
    transform = transforms.Compose(
        [
            transforms.Resize((conf.IMG_SIZE, conf.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    dataset = ImageDataset(conf.Dataset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # Create the model
    start_epoch = 0
    G = Generator(conf.IMG_SIZE, deep_mapping=conf.deep_mapping, latent_size=conf.LATENT)
    D = Discriminator(conf.IMG_SIZE)
    wandb.watch(G)
    wandb.watch(D)

    # Load the pre-trained weight
    if os.path.exists(Weight_dir):
        name_to_epoch = lambda x: int(x.replace('.pth', '').replace('weight ', ''))
        epochs = sorted([name_to_epoch(elem) for elem in os.listdir(Weight_dir) if '.pth' in elem])
        if len(epochs) > 0:
            last_epoch = epochs[-1]
            print(f'Load the pre-trained weight {last_epoch}')
            state = torch.load(f'{Weight_dir}/weight {last_epoch}.pth')
            G.load_state_dict(state['G'])
            D.load_state_dict(state['D'])
            start_epoch = state['start_epoch']

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
        alpha = torch.randn((real_samples.size(0), 1, 1, 1), device=device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Tensor(real_samples.shape[0], 1).fill_(1.0).requires_grad_(False)
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
    for epoch in range(start_epoch, conf.epochs):
        bar = tqdm(dataloader)
        loss_G = []
        loss_D = []
        for i, real_img in enumerate(bar):
            # (1) Update D network
            real_imgs = real_img.to(device)
            optim_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn((real_imgs.size(0), conf.LATENT), device=device)

            # Generate a batch of images
            fake_imgs = G(z)

            # Real images
            real_validity = D(real_imgs)
            # Fake images
            fake_validity = D(fake_imgs.detach())
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + conf.lambda_gp * gradient_penalty

            d_loss.backward()
            optim_D.step()

            loss_D.append(d_loss.item())
            wandb.log({"loss_D": d_loss.item()})

            optim_G.zero_grad()
            # (2) Update G network
            if i % conf.UPD_FOR_GEN == 0:
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                z = torch.randn((real_imgs.size(0), conf.LATENT), device=device)
                fake_imgs = G(z)
                fake_validity = D(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optim_G.step()

                loss_G.append(g_loss.item())
                wandb.log({"loss_G": g_loss.item()})

            # Output training stats
            if (i + 1) % 2 * (i + 1) == 0:
                clear_output(wait=True)
                with torch.no_grad():
                    G.eval()
                    Image = TensorToImage(G(torch.randn(conf.LATENT, device=device)).detach().cpu()[0], 0.5, 0.225)
                    G.train()
                    wandb.log({"Random generated face": wandb.Image(Image)})

            clear_output(wait=True)
            bar.set_description(f"Epoch {epoch + 1}/{conf.epochs} [{i + 1}, {len(dataloader)}]")

        # Save model
        state = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'start_epoch': epoch + 1,
        }
        torch.save(state, f'{Weight_dir}/weight {epoch + 1}.pth')
        wandb.log({"mean_loss_G": np.mean(loss_G), "mean_loss_D": np.mean(loss_D)})


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
