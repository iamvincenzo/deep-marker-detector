import cv2
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torchvision.transforms as transforms

""" Helper function used to plots the gradients flowing 
    through different layers in the net during training. """
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []

    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def plot_imgs(images, outputs):                
    to_pil = transforms.ToPILImage()
    orig_img = np.array(to_pil(images.squeeze(0)))
    recon_img = np.array(to_pil(outputs.squeeze(0)))        
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(orig_img, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Original")
    axes[1].imshow(recon_img, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Reconstructed")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.close()
