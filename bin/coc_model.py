import numpy as np
import cv2
import gzip
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

### ======= Utils.py ======= ###
from bin.utils import dice_coefficient, dice_loss
from bin.utils import plot_performance
### ======================== ###

### ======= Data_Loader.py ======= ###
from bin.dataloaders import COCTrainDataset
### ======================== ###

### ======= Models.py ======= ###
from bin.models import initialize_model
### ======================== ###

class coc_Model:
    def __init__(self, config):

        self.images_path = Path(config['images_path'])
        self.masks_path = Path(config['masks_path'])
        self.path_to_save = Path(config['path_to_save'])
        self.log_path = self.path_to_save
        self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS = config['img_width'], config['img_height'], config['img_channels']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.val_split = config['val_split']
        self.epochs = config['epochs']
        self.patience = config['patience']
        self.TL = config['tl']
        self.weights_path = config['weights_path']

    def train_model(self):
        self.path_to_save.mkdir(exist_ok = True)

        # Load the dataset
        self.dataset = COCTrainDataset(self.images_path, self.masks_path, self.IMG_WIDTH, self.IMG_HEIGHT)
        print("Dataset was loaded...")
        # print("dataset size: ", dataset.size())

        self.train_len = int(len(self.dataset) * (1 - self.val_split))
        self.val_len = len(self.dataset) - self.train_len
        self.train_dataset, self.val_dataset = random_split(self.dataset, [self.train_len, self.val_len])

        print("Train length: ", self.train_len)
        print("Val length: ", self.val_len)

        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

        # Initialize the model, loss function and optimizer
        self.model = initialize_model(self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS).to('cuda')

        if self.TL:
            self.pretrained_weights = torch.load(self.weights_path)
            self.model.load_state_dict(self.pretrained_weights, strict = False)
            self.model.eval()
            print("Weights have been loaded succesfully...")

        print("Model initialized...")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

        # Setup TensorBoard logging
        # writer = SummaryWriter(log_dir=log_path)

        # Early stopping setup
        self.patience_counter = 0
        self.best_val_loss = float('inf')

        self.train_dice_values = []
        self.val_dice_values = []
        self.train_loss_values = []
        self.val_loss_values = []

        print("Training...")
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            self.train_loss = 0.0
            self.train_dice = 0.0
            for X, y in self.train_loader:

                X, y = X.to('cuda'), y.to('cuda')

                self.optimizer.zero_grad()
                self.output = self.model(X)

                # Calculate the Loss
                self.loss = dice_loss(self.output, y)

                self.loss.backward()
                self.optimizer.step()
                self.train_loss += self.loss.item()

                # Calculate the Dice
                self.pred = torch.sigmoid(self.output) > 0.5
                self.train_dice_ = dice_coefficient(self.pred.float(), y)
                self.train_dice += self.train_dice_.item()

            self.train_loss /= len(self.train_loader)
            self.train_dice /= len(self.train_loader)

            # Validate
            self.model.eval()
            self.val_loss = 0.0
            self.val_dice = 0.0
            with torch.no_grad():
                for X, y in self.val_loader:

                    X, y = X.to('cuda'), y.to('cuda')

                    # Calculate the Loss 
                    self.output = self.model(X)
                    # loss = criterion(output, y)
                    self.loss = dice_loss(self.output, y)
                    self.val_loss += self.loss.item()

                    # Calculate the Dice 
                    self.pred = torch.sigmoid(self.output) > 0.5
                    self.dice = dice_coefficient(self.pred.float(), y)
                    self.val_dice += self.dice.item()

            self.val_loss /= len(self.val_loader)
            self.val_dice /= len(self.val_loader)

            self.train_loss_values.append(self.train_loss)
            self.val_loss_values.append(self.val_loss)
            self.train_dice_values.append(self.train_dice)
            self.val_dice_values.append(self.val_dice)

            print(f'Epoch: {epoch + 1}/{self.epochs}, Train Loss: {self.train_loss:.4f}, Train Dice: {self.train_dice:.4f}, Val Loss: {self.val_loss:.4f}, Val Dice: {self.val_dice:.4f}')

            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.path_to_save / 'coc_base_weights.pth')
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    print("Early stopping")
                    break

        self.save_path = os.path.join(self.path_to_save, 'results_coc')
        os.makedirs(self.save_path, exist_ok = True)

        plot_performance(self.train_loss_values, self.val_loss_values, self.train_dice_values, self.val_dice_values, self.save_path)
        print("Figures were saved.")

        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'coc_Final_dict.pt'))