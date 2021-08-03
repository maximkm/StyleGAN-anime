import numpy as np


def TensorToImage(img, mean=0.5, std=0.28):
    # Convert a tensor to an image
    img = np.transpose(img.numpy(), (1, 2, 0))
    img = (img*std + mean)*255
    img = img.astype(np.uint8)    
    return img
