import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import numpy as np
from skimage import transform

def display_image(axis, image_tensor):
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("The `display_image` function expects a `torch.Tensor` " +
                        "use the `ToTensor` transformation to convert the images to tensors.")
        
    # The imshow commands expects a `numpy array` with shape (3, width, height)
    # We rearrange the dimensions with `permute` and then convert it to `numpy`
    image_data = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    height, width, _ = image_data.shape
    axis.imshow(image_data)
    axis.set_xlim(0, width)
    
    # By convention when working with images, the origin is at the top left corner.
    # Therefore, we switch the order of the y limits.
    axis.set_ylim(height, 0)


class Stat:

    def __init__(self):
        self.clear()

    def clear(self):
        self.sum = 0
        self.count = 0

    def __call__(self, val, n=1):
        self.sum += val * n
        self.count += n
    
    def average(self):
        return self.sum/self.count

def visualize_attention(img, alphas, caption, smooth=False):
    """
    param img: image tensor (1, 3, img_dim, img_dim)
    param caption: caption list (1, caption_length)
    param alphas: weights of image (1, num_pixels)
    """

    img = img.squeeze(dim=0).permute(1, 2, 0).cpu().detach().numpy()
    alphas = alphas.unflatten(dim=2, sizes=(14,14)).squeeze(dim=0).cpu().detach().numpy()
    caption = caption[0]

    height, width, _ = img.shape
    plt.figure(figsize=(20,40))

    for t in range(len(caption)):
        plt.subplot(int(np.ceil(len(caption) / 5.)), 5, t + 1)
        plt.text(0, 1, '%s' % (caption[t]), color='black', backgroundcolor='white', fontsize=12)
        
        plt.imshow(img)
        current_alpha = alphas[t]
        if smooth:
            current_alpha = transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            current_alpha = transform.resize(current_alpha, [14 * 21, 14 * 21])

        plt.imshow(current_alpha, alpha = 0.8 if t > 0 else 0)
        #plt.xlim([0, width])
        #plt.ylim([0, height])
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.show()
    return plt.gcf()
