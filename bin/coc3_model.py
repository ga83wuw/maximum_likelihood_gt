import numpy as np
import cv2
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

### ======= Utils.py ======= ###
from bin.utils import dice_coefficient, dice_loss
from bin.utils import noisy_label_loss_GCM, noisy_label_loss_lCM, noisy_loss, noisy_loss2, noisy_loss3, compute_behavior
from bin.utils import plot_performance
from bin.utils import test_lGM
from bin.utils import calculate_cm, evaluate_cm
### ======================== ###

### ======= Data_Loader.py ======= ###
from bin.dataloaders import COC3TrainDataset
### ======================== ###

### ======= Models.py ======= ###
from bin.models import initialize_model_GCM
from bin.models import initialize_model_lCM
### ======================== ###

class coc3_Model:
    def __init__(self, config):

        self.images_path = Path(config['images_path'])
        self.masks_path = Path(config['masks_path'])
        self.path_to_save = Path(config['path_to_save'])
        self.log_path = self.path_to_save
        self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS = config['img_width'], config['img_height'], config['img_channels']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.val_split = config['val_split']
        self.test_split = config['test_split']
        self.epochs = config['epochs']
        self.patience = config['patience']
        self.GCM = config['gcm']
        self.TL = config['tl']
        self.weights_path = config['weights_path']
        self.ALPHA = config['alpha']
        self.DEVICE = config['device']

    def print_matrices(self, model):
        for name, param in self.model.named_parameters():
            if 'cms_output' in name:
                print(param)

    def print_matrices_on_borders(self, borders):
        self.cmA = []
        self.cmB = []
        self.cmC = []
        self.cmDL = []
        for item in borders:
            self.cmA.append(item[0])
            self.cmB.append(item[1])
            self.cmC.append(item[2])
            self.cmDL.append(item[3])

        self.total_cm = [self.cmA, self.cmB, self.cmC, self.cmDL]

        for i, cm in enumerate(self.total_cm):
            self.sum_cm = torch.zeros(2, 2)
            for tensor in cm:
                self.sum_cm += tensor
            self.avrg_cm = self.sum_cm / len(cm)

            if i != 3:
                print("Confusion Matrix of Annotator", (i + 1))
                print(self.avrg_cm)
            else:
                print("Confusion Matrix of DL model")
                print(self.avrg_cm)

    def train_model(self):
        self.path_to_save.mkdir(exist_ok = True)

        # Load the dataset
        self.dataset = COC3TrainDataset(self.images_path, self.IMG_WIDTH, self.IMG_HEIGHT)
        print("Dataset was loaded...")

        self.train_len = int(len(self.dataset) * (1 - self.val_split - self.test_split))
        self.val_len = int(len(self.dataset) * self.val_split)
        self.test_len = len(self.dataset) - self.train_len - self.val_len
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [self.train_len, self.val_len, self.test_len])

        print("Train length: ", self.train_len)
        print("Val length: ", self.val_len)
        print("Test length: ", self.test_len)

        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)
        self.test_loader = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)

        # Initialize the model, loss function and optimizer
        if self.GCM:
            self.model = initialize_model_GCM(self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS).to(self.DEVICE)
            self.noisy_label_loss = noisy_label_loss_GCM
        else:
            self.model = initialize_model_lCM(self.IMG_CHANNELS).to(self.DEVICE)
            self.noisy_label_loss = noisy_label_loss_lCM
            self.test = self.test_lGM
        if self.TL:
            self.pretrained_weights = torch.load(self.weights_path)
            
            ### print names of layers ###
            self.model_dict = self.model.state_dict()
            # for name, param in model.named_children():
            #     print(name)
            self.layers = ['enc1', 'enc2', 'enc3', 'enc4',
                    'middle',
                    'dec4', 'dec3', 'dec2', 'dec1',
                    'upconv4', 'upconv3', 'upconv2', 'upconv1',
                    'cms_output',
                    'output']
            for name, param in self.model.named_parameters():
                if 'cms_output' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            ### ===================== ###

            self.model.load_state_dict(self.pretrained_weights, strict = False)
            self.model.eval()
            print("Weights have been loaded succesfully...")
        
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print("Total number of params: ", self.total_params)
        self.total_params_grad  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total number of params with grad: ", self.total_params_grad)

        print("Model initialized...")
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

        self.scheduler = StepLR(self.optimizer, step_size = 30, gamma = 1.)

        # Setup TensorBoard logging
        # writer = SummaryWriter(log_dir=log_path)

        # Early stopping setup
        self.patience_counter = 0
        self.best_val_loss = float('inf')

        self.train_dice_values = []
        self.val_dice_values = []
        self.train_loss_values = []
        self.train_lossAR_values = []
        self.train_lossHS_values = []
        self.train_lossSG_values = []
        self.val_loss_values = []
        self.val_lossAR_values = []
        self.val_lossHS_values = []
        self.val_lossSG_values = []

        print("Training...")
        self.lossAR, self.lossHS, self.lossSG = 0, 0, 0
        self.borders = []

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            
            self.train_loss = 0.0
            self.train_lossAR = 0.0
            self.train_lossHS = 0.0
            self.train_lossSG = 0.0
            self.train_loss_dice = 0.0
            self.train_loss_trace = 0.0
            self.train_dice = 0.0
            self.cm_AR_true = torch.zeros(2,2)
            self.cm_HS_true = torch.zeros(2,2)
            self.cm_SG_true = torch.zeros(2,2)
            nb=0
            for name, X, y_AR, y_HS, y_SG, y_avrg in self.train_loader:

                X, y_AR, y_HS, y_SG, y_avrg = X.to(self.DEVICE), y_AR.to(self.DEVICE), y_HS.to(self.DEVICE), y_SG.to(self.DEVICE), y_avrg.to(self.DEVICE)

                self.labels_all = []
                self.labels_all.append(y_AR)
                self.labels_all.append(y_HS)
                self.labels_all.append(y_SG)

                self.labels_avrg_all = []
                self.labels_avrg_all.append(y_avrg)
                
                self.names = name

                self.optimizer.zero_grad()

                self.output, self.output_cms = self.model(X)

                self.borders_cm = compute_behavior(pred = self.output, labels = self.labels_all, labels_avrg = self.labels_avrg_all)
                self.borders.append(self.borders_cm)
                
                # Calculate the Loss
                self.loss, self.loss_dice, self.loss_trace = self.noisy_label_loss(self.output, self.output_cms, self.labels_all, self.names, alpha = self.ALPHA)

                self.loss.backward()
                self.optimizer.step()

                self.train_loss += self.loss.item()
                self.train_lossAR += self.lossAR
                self.train_lossHS += self.lossHS
                self.train_lossSG += self.lossSG
                self.train_loss_dice += self.loss_dice.item()
                self.train_loss_trace += self.loss_trace.item()

                # Calculate the Dice
                self.pred = torch.sigmoid(self.output) > 0.5
                self.train_dice_ = dice_coefficient(self.pred.float(), y_avrg)
                self.train_dice += self.train_dice_.item()

            self.train_loss /= len(self.train_loader)
            self.train_lossAR /= len(self.train_loader)
            self.train_lossHS /= len(self.train_loader)
            self.train_lossSG /= len(self.train_loader)
            self.train_loss_dice /= len(self.train_loader)
            self.train_loss_trace /= len(self.train_loader)
            self.train_dice /= len(self.train_loader)

            
            # Validate
            self.model.eval()
            self.val_loss = 0.0
            self.val_lossAR = 0.0
            self.val_lossHS = 0.0
            self.val_lossSG = 0.0
            self.val_loss_dice = 0.0
            self.val_loss_trace = 0.0
            self.val_dice = 0.0
            with torch.no_grad():
                for name, X, y_AR, y_HS, y_SG, y_avrg in self.val_loader:

                    X, y_AR, y_HS, y_SG, y_avrg = X.to(self.DEVICE), y_AR.to(self.DEVICE), y_HS.to(self.DEVICE), y_SG.to(self.DEVICE), y_avrg.to(self.DEVICE)

                    self.labels_all = []
                    self.labels_all.append(y_AR)
                    self.labels_all.append(y_HS)
                    self.labels_all.append(y_SG)

                    self.names = name

                    self.output, self.output_cms = self.model(X)

                    # Calculate the Loss 
                     self.loss, self.loss_dice, self.loss_trace = self.noisy_label_loss(self.output, self.output_cms, self.labels_all, self.names, alpha = self.ALPHA)

                    self.val_loss += self.loss.item()
                    self.val_lossAR += self.lossAR
                    self.val_lossHS += self.lossHS
                    self.val_lossSG += self.lossSG
                    self.val_loss_dice += self.loss_dice.item()
                    self.val_loss_trace += self.loss_trace.item()

                    # Calculate the Dice 
                    self.pred = torch.sigmoid(self.output) > 0.5
                    self.dice = dice_coefficient(self.pred.float(), y_avrg)
                    self.val_dice += self.dice.item()

                self.val_loss /= len(self.val_loader)
                self.val_lossAR /= len(self.val_loader)
                self.val_lossHS /= len(self.val_loader)
                self.val_lossSG /= len(self.val_loader)
                self.val_loss_dice /= len(self.val_loader)
                self.val_loss_trace /= len(self.val_loader)
                self.val_dice /= len(self.val_loader)

            self.train_loss_values.append(self.train_loss)
            self.train_lossAR_values.append(self.train_lossAR)
            self.train_lossHS_values.append(self.train_lossHS)
            self.train_lossSG_values.append(self.train_lossSG)
            self.val_loss_values.append(self.val_loss)
            self.val_lossAR_values.append(self.val_lossAR)
            self.val_lossHS_values.append(self.val_lossHS)
            self.val_lossSG_values.append(self.val_lossSG)
            self.train_dice_values.append(self.train_dice)
            self.val_dice_values.append(self.val_dice)

            print(f'Epoch: {epoch + 1}/{self.epochs}, Train Loss: {self.train_loss:.4f}, Train Loss Dice: {self.train_loss_dice:.4f}, Train Dice: {self.train_dice:.4f}, Val Loss: {self.val_loss:.4f}, Val Dice: {self.val_dice:.4f}')

            self.scheduler.step()

            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.path_to_save / 'coc3_base_weights.pt')
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    print("Early stopping")
                    break

        self.print_matrices_on_borders(borders = self.borders)

        # Test
        with torch.no_grad():
            for name, X, y_AR, y_HS, y_SG, y_avrg in self.test_loader:

                X, y_AR, y_HS, y_SG, y_avrg = X.to(self.DEVICE), y_AR.to(self.DEVICE), y_HS.to(self.DEVICE), y_SG.to(self.DEVICE), y_avrg.to(self.DEVICE)

                self.labels_all = []
                self.labels_all.append(y_AR)
                self.labels_all.append(y_HS)
                self.labels_all.append(y_SG)

                self.names = name

                self.output, self.output_cms = self.model(X)

                # Calculate the Loss 
                self.loss, self.loss_dice, self.loss_trace = self.noisy_label_loss(self.output, self.output_cms, self.labels_all, self.names, alpha = self.ALPHA)

                self.val_loss += self.loss.item()
                self.val_loss_dice += self.loss_dice.item()
                self.val_loss_trace += self.loss_trace.item()

                # Calculate the Dice 
                self.pred = torch.sigmoid(self.output) > 0.5
                self.dice = dice_coefficient(self.pred.float(), y_avrg)
                self.val_dice += self.dice.item()

        self.save_path = os.path.join(self.path_to_save, 'results_coc3')
        os.makedirs(self.save_path, exist_ok = True)
        if self.GCM:
            self.save_path = os.path.join(self.path_to_save, 'global_cm')
            os.makedirs(self.save_path, exist_ok = True)
            if self.TL:
                self.save_path = os.path.join(self.save_path, 'transfer_learning')
                os.makedirs(self.save_path, exist_ok = True)
            else:
                self.save_path = os.path.join(self.save_path, 'random_learning')
        else:
            self.save_path = os.path.join(self.save_path, 'local_cm')
            os.makedirs(self.save_path, exist_ok = True)
            if self.TL:
                self.save_path = os.path.join(self.save_path, 'transfer_learning')
                os.makedirs(self.save_path, exist_ok = True)
            else:
                self.save_path = os.path.join(self.save_path, 'random_learning')
                os.makedirs(self.save_path, exist_ok = True)
            self.test(self.model, self.test_loader, self.noisy_label_loss, self.save_path, self.DEVICE)

        plot_performance(self.train_loss_values, self.val_loss_values, self.train_dice_values, self.val_dice_values, self.save_path)
        plot_performance(self.train_lossAR_values, self.val_lossAR_values, self.train_dice_values, self.val_dice_values, self.save_path, 'AR')
        plot_performance(self.train_lossHS_values, self.val_lossHS_values, self.train_dice_values, self.val_dice_values, self.save_path, 'HS')
        plot_performance(self.train_lossSG_values, self.val_lossSG_values, self.train_dice_values, self.val_dice_values, self.save_path, 'SG')
        print("Figures were saved.")