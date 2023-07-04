import os
import torch
import argparse

import os
import torch
from datetime import datetime
import pytorch_model_summary as pms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from solver import Solver
from custom_dataset import CustomDataset
from custom_dataset import get_dataset


""" Helper function used to get cmd parameters. """
def get_args():
    parser = argparse.ArgumentParser()

    # model-infos
    #######################################################################################
    parser.add_argument("--run_name", type=str, default="run_1",
                        help="the name assigned to the current run")

    parser.add_argument("--model_name", type=str, default="unet",
                        help="the name of the model to be saved or loaded")
    #######################################################################################

    # model-types
    #######################################################################################
    parser.add_argument("--select_model", type=str, default="YOLO",
                        choices=["UNet", "YOLO", "AutoEncoder"], help="select the model to train")

    parser.add_argument("--pretrained", action="store_true",
                        help="indicates whether to load a pretrained model")

    parser.add_argument("--freeze", action="store_true",
                        help="indicates whether to freeze a pretrained model")
    #######################################################################################

    # network-architecture-parameters
    #######################################################################################
    # num of filters in each convolutional layer (num of channels of the feature-map): 
    # so you can modify the network architecture
    parser.add_argument('--features', nargs='+',  default=[64, 128, 256, 512],
                        help='a list of feature values (number of filters i.e., neurons in each layer)')
    
    parser.add_argument("--use_batch_norm", action="store_true",
                        help="indicates whether to use batch normalization layers in each conv layer")

    parser.add_argument("--weights_init", action="store_true",
                        help="determines whether to use weights initialization")
    #######################################################################################

    # training-parameters (1)
    #######################################################################################
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="the total number of training epochs")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="the batch size for training and validation data")
  
    parser.add_argument("--workers", type=int, default=4,
                        help="the number of workers in the data loader")
    #######################################################################################

    # training-parameters (2)
    #######################################################################################
    parser.add_argument("--random_seed", type=int, default=42,
                        help="the random seed used to ensure reproducibility")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="the learning rate for optimization")

    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse"],
                        help="the loss function used for model optimization")

    parser.add_argument("--opt", type=str, default="Adam", 
                        choices=["SGD", "Adam"],
                        help="the optimizer used for training")

    parser.add_argument("--patience", type=int, default=5,
                        help="the threshold for early stopping during training")
    #######################################################################################

    # training-parameters (3)
    #######################################################################################
    parser.add_argument("--load_model", action="store_true",
                        help="determines whether to load the model from a checkpoint")

    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints", 
                        help="the path to save the trained model")
    #######################################################################################

    # data-path
    #######################################################################################
    parser.add_argument("--dataset_path", type=str, default="./data",
                        help="path where to save/get the dataset")
    #######################################################################################

    # data transformation
    #######################################################################################
    parser.add_argument("--norm_input", action="store_true",
                        help="indicates whether to normalize the input data")
    
    parser.add_argument("--apply_transformations", action="store_true",
                        help="indicates whether to apply transformations to images")
    #######################################################################################

    return parser.parse_args()

""" https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py """
""" Helper function used to compute mean and standard deviation of the training set. """
def get_mean_std(dataloader):
    print(f"\nStarting to compute the mean and standard deviation of the training set...")  

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for _, (data, _) in enumerate(dataloader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    print("\nmean: ", mean)
    print("std: ", std)

    print(f"\nMean and stdandard deviation computation Done...")

    return mean, std

""" Helper function used to select the model to train. """
def model_selection(args, input_size, num_classes):
    pass

""" Main function used to run the experiment. """
def main(args):
    # tensorboard specifications
    date =  "_" + datetime.now().strftime("%d%m%Y-%H%M%S")
    writer = SummaryWriter("./runs/" + args.run_name + date)
    
    # custom dataset
    img_files_train, img_files_valid = get_dataset()

    if args.norm_input:
        # compute mean and std of unormalized data
        dataset = CustomDataset(img_files_train, args, normalize=None, train=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        mean, std = get_mean_std(dataloader)
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        normalize, mean, std = None, None, None

    train_dataset = CustomDataset(img_files_train, args, normalize=normalize, train=True)
    valid_dataset = CustomDataset(img_files_valid, args, normalize=normalize, train=False)

    # pin_memory: speed up the host (cpu) to device (gpu) transfer
    pin = True if torch.cuda.is_available() else False

    # DataLoader wraps an iterable around the Dataset to enable easy access to the samples
    # according to a specific batch-size (load the data in memory)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.workers, pin_memory=pin) 
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.workers, pin_memory=pin)

    # # cuDNN supports many algorithms to compute convolution:
    # # autotuner runs a short benchmark and selects the algorithm with the best performance
    torch.backends.cudnn.benchmark = True

    # select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: ", device)

    # get input shape
    inputs, _ = next(iter(trainloader))
    input_size = inputs.shape[1:]
    # get output shape 
    num_classes = args.num_classes

    # get the model
    model = model_selection(args=args, input_size=input_size, num_classes=num_classes)

    print("\nModel: ")
    pms.summary(model, torch.zeros(inputs.shape), max_depth=5, print_summary=True)

    # early-stopping-patience: 10% of total epochs
    args.patience = int(0.1 * args.num_epochs)

    # define solver class instance
    solver = Solver(train_loader=trainloader, valid_loader=validloader, 
                    model=model, device=device, writer=writer, 
                    normalize=(mean, std), args=args)
    
    solver.train_model()

""" Starting the simulation. """
if __name__ == "__main__":
    args = get_args()
    
    # if folder doesn't exist, then create it
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.isdir("statistics"):
        os.makedirs("statistics")
    
    print(f"\n{args}")
    main(args)