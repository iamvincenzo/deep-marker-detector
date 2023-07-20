import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

""" Function used to generate a proportioned dataset: 70% training and 30% validation. """
def get_dataset(dataset_path, random_seed):
    random.seed(random_seed)

    img_files_train = []
    mask_files_train = []   
    img_files_valid = [] 
    mask_files_valid = []
        
    split_percentage = 0.7
    
    image_paths = os.listdir(os.path.join(dataset_path, "images"))

    # randomly select elements for train and validation sets
    train_elements = random.sample(image_paths, int(len(image_paths) * split_percentage))
    test_elements = list(set(image_paths) - set(train_elements))

    img_files_train += [os.path.join(dataset_path, "images", elem) for elem in train_elements]
    img_files_valid += [os.path.join(dataset_path, "images", elem) for elem in test_elements]

    mask_files_train = [file.replace("images", "masks") for file in img_files_train]
    mask_files_valid = [file.replace("images", "masks") for file in img_files_valid]

    print(f'\nNumber of elements in img_files_train: {len(img_files_train)}')
    print(f'Number of elements in img_files_valid: {len(img_files_valid)}')

    # # debug: check for duplicates
    # print(f"\nCheck duplicates in train/test set: {list(set(img_files_train).intersection(img_files_valid))}\n")

    return img_files_train, mask_files_train, img_files_valid, mask_files_valid

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
