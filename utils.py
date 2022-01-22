import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import numpy as np
from skimage import transform
import torchvision.transforms as T


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

def visualize_attention(img, alphas, caption, invert_normalization = True):

    if invert_normalization:
        inv_normalize = T.Normalize( mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.255])
        img = inv_normalize(img)

    img = img.squeeze(dim=0).permute(1, 2, 0).cpu().detach().numpy()
    alphas = alphas.unflatten(dim=1, sizes=(14,14)).cpu().detach().numpy()

    height, width, _ = img.shape
    num_cols = 5
    num_rows = int(np.ceil(len(caption) / float(num_cols)))
    img_size = 4
    plt.figure(figsize=(num_cols * img_size ,num_rows * img_size))

    for t in range(len(caption)):
        plt.subplot(num_rows, num_cols, t + 1)
        plt.text(0, 20, '%s' % (caption[t]), color='black', backgroundcolor='white', fontsize=12)
        
        plt.imshow(np.clip(img, 0, 1))
        current_alpha = alphas[t]
        current_alpha = transform.resize(current_alpha, [14 * 21, 14 * 21])

        plt.imshow(current_alpha, alpha = 0.8 if t > 0 else 0)
        #plt.xlim([0, width])
        #plt.ylim([0, height])
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.show()
    return plt.gcf()