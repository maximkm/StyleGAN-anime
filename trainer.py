from IPython.display import clear_output
from utils import register, images
from torch.autograd import grad
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import wandb


trainers = register.ClassRegistry()


@trainers.add_to_registry(name="base")
class BaseGANTrainer:
    def __init__(self, conf, **kwargs):
        '''
        Input:
            conf: dictionary of training settings
            **kwargs: the dictionary must contain:
                "G": initialized generator
                "D": initialized discriminator
                "start_epoch": continuing training
                "dataloader": dataset loader
                "optim_G": generator optimizer
                "optim_D": discriminator optimizer
                "gen_loss": generator loss function
                "disc_loss":  discriminator loss function
                "z_dim": the dimension of the generator vector
                "device": contains a device type
        '''
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.conf = conf
        if self.conf['wandb_set']:
            wandb.login()
            wandb.init(
                project=self.conf['Generator'],
                config=self.conf
            )
            wandb.watch(self.G)
            wandb.watch(self.D)
          

    def logger(self, data):
        if self.conf['wandb_set']:
            wandb.log(data)
        
  
    def save_model(self, epoch):
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'start_epoch': epoch + 1,
        }
        torch.save(state, f'{self.conf["Weight_dir"]}/weight {epoch + 1}.pth')


    def generate_images(self, cnt=1):
        # Sample noise as generator input
        z = torch.randn((cnt, self.z_dim), device=self.device)
        return self.G(z)
        
    
    def train_disc(self, real_imgs, fake_imgs):
        real_logits = self.D(real_imgs)
        fake_logits = self.D(fake_imgs)
        return self.disc_loss(real_logits, fake_logits)
        
        
    def train_gen(self, fake_imgs):
        logits_fake = self.D(fake_imgs)
        return self.gen_loss(logits_fake)
    
    
    def train_loop(self):
        warnings.filterwarnings("ignore")
        for epoch in range(self.start_epoch, self.conf['epochs']):
            bar = tqdm(self.dataloader)
            loss_G, loss_D = [], []
            for i, real_img in enumerate(bar):
                self.G.zero_grad()
                real_imgs = real_img.to(self.device)
                
                # Generate a batch of images
                fake_imgs = self.generate_images(real_imgs.size(0)).detach()

                # Update D network
                d_loss = self.train_disc(real_imgs, fake_imgs)
                
                self.D.zero_grad()
                d_loss.backward()
                self.optim_D.step()

                loss_D.append(d_loss.item())
                self.logger({"loss_D":d_loss.item()})

                if i % self.conf['UPD_FOR_GEN'] == 0:
                    # Generate a batch of images
                    fake_imgs = self.generate_images(real_imgs.size(0))

                    # Update G network
                    g_loss = self.train_gen(fake_imgs)
                    
                    self.G.zero_grad()
                    g_loss.backward()
                    self.optim_G.step()

                    loss_G.append(g_loss.item())
                    self.logger({"loss_G":g_loss.item()})

                # Output training stats
                if i % 5 == 0:
                    clear_output(wait=True)
                    with torch.no_grad():
                        self.G.eval()
                        Image = images.TensorToImage(self.generate_images().detach().cpu()[0], 0.5, 0.225)
                        self.logger({"Random generated face": wandb.Image(Image)})
                        self.G.train()

                clear_output(wait=True)        
                bar.set_description(f"Epoch {epoch + 1}/{self.conf['epochs']} D_loss: {round(loss_D[-1], 2)} G_loss: {round(loss_G[-1], 2)}")

            # Save model
            self.save_model(epoch)
            self.logger({"mean_loss_G":np.mean(loss_G), "mean_loss_D":np.mean(loss_D)})


@trainers.add_to_registry(name="gp")            
class GpGANTrainer(BaseGANTrainer):
    def __init__(self, conf, **kwargs):
        super().__init__(conf, **kwargs) 
        
        
    def train_disc(self, real_imgs, fake_imgs):
        lambda_gp = self.conf["Loss_config"]["lambda_gp"]
        return self.disc_loss(self.D, real_imgs, fake_imgs, lambda_gp)
