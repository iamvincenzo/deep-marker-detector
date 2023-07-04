import os
import math
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

""" Function used to generate a proportioned dataset: 70% training and 
    30% validation per each class. """
def get_dataset(dataset_path, th, random_seed):
    pass
    # return img_files_train, mask_files_train, img_files_valid, mask_files_valid


""" Custom class used to create the training and test sets. """
class CustomDataset(Dataset):
    """ Initialize configurations. """
    def __init__(self, image_paths, target_paths, args, normalize=None, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.args = args
        self.train = train
        self.normalize = normalize

    """ Method used to apply transformation to images 
        and their corresponding masks."""
    def transform(self, image, mask):
         # transformation applied only if required and only to training images
        if self.args.apply_transformations and self.train:
            # random horizontal flipping (we apply transforms here because we need to apply
            # them with the same probability to both img and mask)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # random vertical flip
            if random.random() > 0.3:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # # random rotation
            # if random.random() > 0.4:
            #     angle = random.randint(-30, 30)
            #     image = TF.rotate(image, angle)
            #     mask = TF.rotate(mask, angle)

        # to tensor and remove the alpha channel if present (PNG format)
        trnsf = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x[:3])])
        image = trnsf(image)
        mask = trnsf(mask)

        # input normalization if required
        if self.normalize is not None:
            image = self.normalize(image)

        return image, mask

    """ Method used to get (image, mask). """
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        mask_path = self.target_paths[index]
        mask = Image.open(mask_path)
        x, y = self.transform(image, mask)

        return x, y

    def __len__(self):
        return len(self.image_paths)
