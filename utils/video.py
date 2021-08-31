from IPython.display import clear_output, HTML, display
from matplotlib import animation, pyplot as plt


def GenerateVideo(samples_list, delay=1000):
    def process_image(img):
        if len(img.shape) == 4:
            img = img[0]
        plt.axis('off')
        return [plt.imshow(TensorToImage(img, 0.5, 0.28))]

    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    frames = [process_image(elem) for elem in samples_list]
    
    ani = animation.ArtistAnimation(fig, frames, interval=delay)  
    display(HTML(ani.to_html5_video()))
