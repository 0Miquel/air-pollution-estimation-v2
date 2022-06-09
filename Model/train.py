import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models, transforms
from dataset import ChinaDataset
from torch.utils.data import DataLoader
import time
import copy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import argparse
import sys
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from pytorch_forecasting.metrics import MAPE, SMAPE
import math
from model import MyModel
from torch.optim.lr_scheduler import MultiStepLR

def get_dataloaders(batch_size, locations, train_ratio):
    dataset = ChinaDataset("../Datasets/FinalImages/", locations, "../Datasets/GT/finalData/gt_data.csv", "../Datasets/Weather/finalData/weather_data.csv")
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(test_dataset, batch_size=batch_size)
    }
    return dataloaders


def get_model(tabular, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)

    model = MyModel(model_ft, tabular)
    #print(model)
    return model.cuda()


def get_optimizer(optimizer, model, lr):
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[33, 66], gamma=0.5)
    return optimizer, scheduler


def get_criterion(criterion_name):
    if criterion_name == 'MSE':
        # Mean Square Error
        criterion = nn.MSELoss()
    elif criterion_name == 'MAE':
        # Mean Absolute Error
        criterion = nn.L1Loss()
    return criterion


def get_smape(gt, preds):
    smape = SMAPE()  # metric
    smape_preds = smape.loss(torch.FloatTensor(preds), torch.FloatTensor(gt))
    mean_smape = torch.mean(smape_preds).item()
    return mean_smape


def train(model, dataloaders, optimizer, scheduler, criterion, num_epochs):
    since = time.time()

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            total_gt = []
            total_outputs = []
            running_loss = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to validation mode
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                # Iterate over data.
                for images, data, label, gt in tepoch: #image, tabular data, label classification, ground truth value in ug/m3
                    tepoch.set_description(f"Epoch {epoch} {phase}")
                    images = images.float().cuda()
                    data = data.float().cuda()
                    #transpose the flat gt tensor
                    gt0 = gt.view(gt.shape[0], 1).float().cuda()
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images, data)
                        #flatten outputs
                        outputs0 = outputs.view(1, outputs.shape[0])[0]
                        loss = criterion(outputs, gt0)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        total_gt = total_gt + gt.tolist()
                        total_outputs = total_outputs + outputs0.tolist()
                    # Loss
                    running_loss += loss.item() * images.size(0)
                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    smape = get_smape(total_gt, total_outputs)

                    tepoch.set_postfix(loss=epoch_loss, smape=smape)

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tabular', default='True')
    args = parser.parse_args(sys.argv[1:])

    tabular = args.tabular == 'True'

    locations = ['Beijing', 'Shanghai']
    criterion_name = 'MAE'
    optimizer = 'Adam'
    train_ratio = 0.7
    batch_size = 8
    lr = 0.001
    epochs = 100


    model = get_model(tabular)
    dataloaders = get_dataloaders(batch_size=batch_size, locations=locations, train_ratio=train_ratio)
    criterion = get_criterion(criterion_name)
    opt, scheduler = get_optimizer(optimizer, model, lr)

    train(model, dataloaders, opt, scheduler, criterion, epochs)
