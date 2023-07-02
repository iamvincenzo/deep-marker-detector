import os
import math
import torch
import random
import glob as glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def get_dataset(args):
    pass

""" Custom class used to create the training, validation and test sets. """
class CustomDataset(Dataset):    
    """ Initialize configurations. """
    def __init__(self, image_paths, args, normalize=None, train=True):
        super().__init__()
        self.args = args
        self.train = train
        self.image_paths = image_paths
        self.normalize = normalize

    """ Method used to apply transformation to images. """
    def transform(self, image):
        # transformation applied only if required and only to training images
        if self.args.apply_transformations and self.train:
            # random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
            # random vertical flip
            if random.random() > 0.3:
                image = TF.vflip(image)
            # random rotation
            if random.random() > 0.4:
                angle = random.randint(-30, 30)
                image = TF.rotate(image, angle)

        # to tensor and remove the alpha channel if present (PNG format)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x[:3])])
        image = transform(image)

        # input normalization if required
        if self.normalize is not None:
            image = self.normalize(image)

        return image

    """ Method used to get (image, label). """
    def __getitem__(self, index):
        # image_paths = (path, label)
        image_path = self.image_paths[index][0]
        label = self.image_paths[index][1]
        image = Image.open(image_path)
        x = self.transform(image)

        return x, label

    """ Method used to get the dataset lenght. """
    def __len__(self):
        return len(self.image_paths)