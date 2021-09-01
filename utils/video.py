from IPython.display import HTML, display
from matplotlib import animation, pyplot as plt
from utils.images import TensorToImage


def GenerateVideo(samples_list, delay=1000, mean=0.5, std=0.375):
    def process_image(img):
        if len(img.shape) == 4:
            img = img[0]
        plt.axis('off')
        return [plt.imshow(TensorToImage(img, mean, std))]

    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    frames = [process_image(elem) for elem in samples_list]
    
    ani = animation.ArtistAnimation(fig, frames, interval=delay)  
    display(HTML(ani.to_html5_video()))
