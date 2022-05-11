import torchvision.models as models
import torch
import torch.nn.functional
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import cv2
import numpy as np
from sklearn.metrics import classification_report

def my_cv_imread(filepath):
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img
 
 
# 图像处理
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
])
dataset = ImageFolder(r'F:\PersonalFiles\CourseMaterials\大二下\机器学习\Animals-10', transform=transform)
 
train = int(len(dataset) * 0.8)
other_train = len(dataset) - train
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train, other_train])
 
train_dataloader_train = DataLoader(train_dataset, batch_size=1024, shuffle=True)
train_dataloader_test = DataLoader(test_dataset, batch_size=1024, shuffle=True)

# vgg-net
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        for name, value in model.named_parameters():
            value.requires_grad = False
        self.vgg_layer = nn.Sequential(*list(model.children())[:-2])
        # first layer
        self.Linear_layer1 = nn.Linear(512, 4096)
        # second layer
        self.Linear_layer2 = nn.Linear(4096, 512)
        # third layer
        self.Linear_layer3 = nn.Linear(512, 10)
        # drop layer
        self.drop_layer = torch.nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.vgg_layer(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = torch.nn.functionalF.relu(self.Linear_layer1(x))
        x = self.drop_layer(x)
        x = torch.nn.functional.relu(self.Linear_layer2(x))
        x = self.drop_layer(x)
        x = self.Linear_layer3(x)
        return x