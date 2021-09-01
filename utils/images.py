from PIL import Image
from tqdm import tqdm
import numpy as np
import os


def TensorToImage(img, mean=0.5, std=0.375):
    # Convert a tensor to an image
    img = np.transpose(img.numpy(), (1, 2, 0))
    img = (img*std + mean)*255
    img = img.astype(np.uint8)    
    return img


def SaveImages(Trainer, dir='img', cnt=1, mean=0.5, std=0.375):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for n in tqdm(range(cnt)):
        img = Trainer.generate_images()[0]
        image = Image.fromarray(TensorToImage(img.detach().cpu(), mean, std))
        image.save(f'{dir}/{n}.png')
