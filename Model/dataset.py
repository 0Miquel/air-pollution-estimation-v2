import os
import json
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from torchvision import models, transforms
import random
import pandas as pd

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

t = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.Resize((224, 224))
    ])

"""
pytorch dataset
dataset structure:
image_path, location, year, month, day, hour, Air quality, pollutant_measure
"""

class ChinaDataset(Dataset):
    def __init__(self, root_path, locations, gt_path, weather_path):
        super().__init__()
        self.gt_path = gt_path
        self.root_path = root_path
        self.locations = locations

        #read image data
        img_data = []
        for location in self.locations:
            files = os.listdir(self.root_path+location)
            img_data = img_data + [[self.root_path + location + "/" + file, location, int(file[:4]), int(file[4:6]), int(file[6:8]), int(file[9:11])] for file in files]
        df_img = pd.DataFrame(img_data, columns=['Path', 'Site', 'Year', 'Month', 'Day', 'Hour'])

        #read ground truth data
        df_gt = pd.read_csv(gt_path)
        #inner join between image and ground truth data
        df_merged = pd.merge(df_img, df_gt, on=["Site", "Year", 'Month', 'Day', 'Hour'])

        df_weather = pd.read_csv(weather_path)
        # inner join between image and ground truth data
        df_merged = pd.merge(df_weather, df_merged, on=["Site", "Year", 'Month', 'Day'])

        df_merged = df_merged[["Path", "Site", "Year", "Month", "Day", "Hour", "AQI Category", "Raw Conc.", "temp",
                               "humidity", "precip", "sealevelpressure", "windspeed", "winddir", "cloudcover", "uvindex"]]

        #save dataset as list of lists, it is easier to access with the getitem method
        self.dataset = df_merged.values.tolist()


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        data = self.dataset[idx]
        img_path = data[0]
        classification = data[6]
        value = data[7]
        meteo_data = data[8:] #["temp", "humidity", "precip", "sealevelpressure", "windspeed", "winddir", "cloudcover", "uvindex"]

        img = cv2.imread(img_path)[:, :, ::-1] #convert bgr to rgb
        #imagenet standarization and resnet resize
        input = t(img.copy())

        return input, torch.FloatTensor(meteo_data), classification, value