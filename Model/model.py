import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
import numpy as np


class MyModel(nn.Module):
    def __init__(self, model, tabular):
        super(MyModel, self).__init__()
        self.cnn = model

        self.tabular = tabular
        if self.tabular:
            # 8 tabular features + image feature
            self.fc = nn.Linear(8 + 1, 1)

    def forward(self, image, data):
        x = self.cnn(image)

        if self.tabular:
            x1 = data
            x = torch.cat((x, x1), dim=1)
            x = self.fc(x)
        return x
