import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import StepLR
# from torch.optim.lr_scheduler import ReduceLROnPlateau

from metrics import dc_loss
from metrics import jac_loss
from plotting_utils import plot_imgs
from pytorchtools import EarlyStopping
# from plotting_utils import plot_grad_flow


""" Solver for training, validation and testing. """
class Solver(object):
    """ Initialize configurations. """
    def __init__(self, train_loader, valid_loader, model, device, writer, normalize, args): 
        self.args = args
        self.model_name = f"{self.args.model_name}_model.pt"
        self.num_epochs = self.args.num_epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.patience = self.args.patience
        self.writer = writer
        self.device = device
        self.start_epoch = 0
        self.mean, self.std = normalize
        self.step = 0

        self.model = model.to(device)

        if args.loss == "bcewll":
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss == "dc_loss":
            self.criterion = dc_loss
        else:
            self.criterion = jac_loss

        if self.args.opt == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, 
                                       momentum=0.9, weight_decay=0.0005)
        elif self.args.opt == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.args.lr, betas=(0.9, 0.999))
        
        # # modify the learning rate during the training process
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
        #                                    factor=0.1, patience=5, verbose=True)
        # # self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        
        if self.args.load_model:
            self.load_model()

    """ Method used to load the model. """
    def load_model(self):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        checkpoint = torch.load(check_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        
        print("\nModel loaded...")
    
    """ Method used to unormalize images. """
    def inverse_normalize(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        
        return tensor

    """ Method used to train the model with an early-stopping implementation. """
    def train_model(self):
        print("\nStarting training...")

        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = []

        # # define the figure where to plot grads info
        # grads_fig = plt.figure(figsize=(12, 6))

        # initialize the early_stopping object
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        early_stopping = EarlyStopping(patience=self.patience, 
                                       verbose=True, path=check_path)

        # put the model in training mode
        self.model.train()

        # loop over the dataset multiple times
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.num_epochs}]")

            # used for creating a terminal progress bar
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)

            for batch, (images, mask) in loop:
                # put data on correct device
                images, mask = images.to(self.device), mask.to(self.device)

                # clear the gradients of all optimized variables   
                self.optimizer.zero_grad()
                
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = self.model(images)

                # calculate the loss
                loss = self.criterion(outputs, mask)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # # gradient clipping to avoid exploding gradients
                # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)

                # # plots the gradients flowing through different layers in the net during training
                # plot_grad_flow(self.model.named_parameters()
                
                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # record training loss
                train_losses.append(loss.item())

            # validate the model at the end of each epoch
            self.validate_model(epoch, valid_losses)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # # update the scheduler with the metric
            # self.scheduler.step(valid_loss)            
            # lr_train = self.optimizer.state_dict()['param_groups'][0]['lr']
            # # update the learning rate scheduler
            # self.scheduler.step()
            # lr_train = self.scheduler.get_last_lr()
            
            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.num_epochs}] | train-loss: {train_loss:.4f}, "
                  f"validation-loss: {valid_loss:.4f} ") # | lr: {lr_train}")
            
            self.writer.add_scalar("training-loss", train_loss, epoch * len(self.train_loader) + batch)
            self.writer.add_scalar("validation-loss", valid_loss, epoch * len(self.valid_loader) + batch)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(epoch, self.model, self.optimizer, valid_loss)

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                break

        print("\nTraining model Done...\n")

        # write all remaining data in the buffer
        self.writer.flush()
        # free up system resources used by the writer
        self.writer.close() 
        
        # # show grad flow
        # plt.tight_layout()
        # plt.show()        

    """ Method used to evaluate the model on the validation/test set. """
    def validate_model(self, epoch, valid_losses):
        print(f"\nEvaluation iteration | Epoch [{epoch + 1}/{self.num_epochs}]")
        
        # put model into evaluation mode
        self.model.eval()

        # no need to calculate the gradients for our outputs
        with torch.no_grad():
            loop = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), leave=True)
            
            for _, (images, mask) in loop:
                # put data on correct device
                images, mask = images.to(self.device), mask.to(self.device)
   
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = self.model(images)
                
                # calculate losses
                loss = self.criterion(outputs, mask)

                valid_losses.append(loss.item())

                if self.std != None:
                    plot_imgs(self.inverse_normalize(images[0]), outputs[0])
                else:
                    plot_imgs(images[0], outputs[0])

                img_grid_fake = torchvision.utils.make_grid(images[:16], normalize=True)
                img_grid_real = torchvision.utils.make_grid(outputs[:16], normalize=True)

                self.writer.add_image("Images", img_grid_fake, global_step=self.step)
                self.writer.add_image("Prediction mask", img_grid_real, global_step=self.step)

                self.step += 1
                
        # reput model into training mode
        self.model.train()

    """ Method used to train the model with an early-stopping implementation. """
    def train_model_1(self):
        print("\nStarting training 1...")

        self.criterion = jac_loss

        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = []

        # initialize the early_stopping object
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        early_stopping = EarlyStopping(patience=self.patience, 
                                       verbose=True, path=check_path)

        # put the model in training mode
        self.model.train()

        # loop over the dataset multiple times
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.num_epochs}]")

            # used for creating a terminal progress bar
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)

            for batch_id, batch_samples in loop:
                print(F"bacth-id: {batch_id}")
                # clear the gradients of all optimized variables   
                self.optimizer.zero_grad()
                loss = 0
                
                for images, mask in batch_samples:
                    # put data on correct device
                    images, mask = images.unsqueeze(0).to(self.device), mask.unsqueeze(0).to(self.device)
                                 
                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs = self.model(images)

                    # calculate the loss
                    loss += self.criterion(outputs, mask)

                loss /= len(batch_samples)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # record training loss
                train_losses.append(loss.item())                    

                # perform a single optimization step (parameter update)
                self.optimizer.step()                   

            # validate the model at the end of each epoch
            self.validate_model_1(epoch, valid_losses)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.num_epochs}] | train-loss: {train_loss:.4f}, "
                  f"validation-loss: {valid_loss:.4f}")
            
            self.writer.add_scalar("training-loss", train_loss, epoch * len(self.train_loader) + batch_id)
            self.writer.add_scalar("validation-loss", valid_loss, epoch * len(self.valid_loader) + batch_id)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(epoch, self.model, self.optimizer, valid_loss)

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                break

        print("\nTraining model Done...\n")

        # write all remaining data in the buffer
        self.writer.flush()
        # free up system resources used by the writer
        self.writer.close() 

    """ Method used to evaluate the model on the validation/test set. """
    def validate_model_1(self, epoch, valid_losses):
        print(f"\nEvaluation iteration | Epoch [{epoch + 1}/{self.num_epochs}]")
        
        # put model into evaluation mode
        self.model.eval()

        # no need to calculate the gradients for our outputs
        with torch.no_grad():
            loop = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), leave=True)

            for _, batch_samples in loop:
                loss = 0

                for images, mask in batch_samples:
                    # put data on correct device
                    images, mask = images.unsqueeze(0).to(self.device), mask.unsqueeze(0).to(self.device)            
                
                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs = self.model(images)

                    img_grid_fake = torchvision.utils.make_grid(images[:16], normalize=True)
                    img_grid_real = torchvision.utils.make_grid(outputs[:16], normalize=True)

                    self.writer.add_image("Images", img_grid_fake, global_step=self.step)
                    self.writer.add_image("Prediction mask", img_grid_real, global_step=self.step)

                    self.step += 1
                    
                    # calculate losses
                    loss += self.criterion(outputs, mask)

                loss /= len(batch_samples)
                
                valid_losses.append(loss.item())
                
        # reput model into training mode
        self.model.train()
