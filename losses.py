import torch
import torch.nn.functional as F
from utils import register


gen_losses = register.ClassRegistry()
disc_losses = register.ClassRegistry()


@disc_losses.add_to_registry("bce")
def binary_cross_entopy(logits_real, logits_fake):
    labels_real = torch.ones_like(logits_real)
    loss = F.binary_cross_entropy_with_logits(logits_real, labels_real)
    labels_fake = torch.zeros_like(logits_fake)
    loss += F.binary_cross_entropy_with_logits(logits_fake, labels_fake)
    return loss


@gen_losses.add_to_registry("st_bce")
def saturating_bce_loss(logits_fake):
    zeros_label = torch.zeros_like(logits_fake)
    loss = -F.binary_cross_entropy_with_logits(logits_fake, zeros_label)
    return loss


@disc_losses.add_to_registry("hinge")
def disc_hinge_loss(logits_real, logits_fake):
    loss = F.relu(1.0 - logits_real).mean()
    loss += F.relu(1.0 + logits_fake).mean()
    return loss


@gen_losses.add_to_registry("hinge")
def gen_hinge_loss(logits_fake):
    loss = -logits_fake.mean()
    return loss


@disc_losses.add_to_registry("wgan")
def disc_wgan_loss(logits_real, logits_fake):
    loss = -logits_real.mean() + logits_fake.mean()
    return loss


@gen_losses.add_to_registry("wgan")
def gen_wgan_loss(logits_fake):
    loss = -logits_fake.mean()
    return loss


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


@disc_losses.add_to_registry("wgan-gp")
def disc_wgan_gp_loss(D, real_imgs, fake_imgs, lambda_gp):
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)
    # Real images
    real_validity = D(real_imgs)
    # Fake images
    fake_validity = D(fake_imgs.detach())
    # Adversarial loss
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    
    
@gen_losses.add_to_registry("wgan-gp")
def gen_wgan_gp_loss(logits_fake):
    loss = -logits_fake.mean()
    return loss
