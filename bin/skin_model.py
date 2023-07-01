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
from bin.dataloaders import SkinTrainDataset
### ======================== ###

### ======= Models.py ======= ###
from bin.models import initialize_model
### ======================== ###

images_path = Path("/data/eurova/cumulus_database/numpy/melanoma/imgs_train.npy.gz")
masks_path = Path("/data/eurova/cumulus_database/numpy/melanoma/imgs_mask_train.npy.gz")
path_to_save = Path("/data/eurova/multi_annotators_project/LNLMI/Results/skin/skin_tf/")
log_path = Path("/data/eurova/multi_annotators_project/LNLMI/Results/skin/skin_tf/")

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 192, 240, 1
learning_rate = 1e-3
batch_size = 16
val_split = 0.1
epochs = 100
patience = 500

class skin_Model:
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

    def train_model(self):
        self.path_to_save.mkdir(exist_ok = True)

        # Load the dataset
        self.dataset = SkinTrainDataset(self.images_path, self.masks_path, self.IMG_WIDTH, self.IMG_HEIGHT)
        print("Dataset was loaded...")
        # print("dataset size: ", dataset.size())

        self.train_len = int(len(self.dataset) * (1 - self.val_split))
        self.val_len = len(self.dataset) - self.train_len
        self.train_dataset, self.val_dataset = random_split(self.dataset, [self.train_len, self.val_len])

        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

        # Initialize the model, loss function and optimizer
        self.model = initialize_model(self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS).to('cuda')
        print("Model initialized...")
        self.criterion = nn.BCEWithLogitsLoss(reduce = 'mean')  # The loss function
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
                self.loss = self.criterion(self.output, y)
                # loss = dice_loss(output, y)

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
                    self.loss = self.criterion(self.output, y)
                    # loss = dice_loss(output, y)
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

            # TensorBoard logging
            # writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

            # Check for early stopping
            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.path_to_save / 'melanoma_base_weights.pth')
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    print("Early stopping")
                    break

        self.save_path = os.path.join(self.path_to_save, 'results_skin')
        os.makedirs(self.save_path, exist_ok = True)

        plot_performance(self.train_loss_values, self.val_loss_values, self.train_dice_values, self.val_dice_values, self.save_path)
        print("Figures were saved.")

        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'skin_Final_dict.pt'))

        # Save the training history
        # np.save(str(path_to_save / 'melanoma_base_history_.npy'), {'train_loss': train_loss, 'val_loss': val_loss})