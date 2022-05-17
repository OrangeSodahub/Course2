from numpy import size
import torch
import torch.nn as nn

# Build AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # first layer: conv {3,227,227} -> {96,55,55} -> {96,27,27} -> {256,27,27} -> {256,13,13}
        self.feature_extraction = nn.Sequential(
            # {3,227,227} -> {96,55,55}
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        # second layer: full
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
import cv2
import os

# Create transform to transform img-data-type to pytorch-tensor
# normalize to [0.0 1.0]
data_transform = transforms.Compose([
    transforms.ToTensor()
])

class AnimalDataSet(Data.Dataset):
    def __init__(self, mode, dir):
        self.labels = ['butterfly','cat','cow','chicken','dog','elephant','horse','sheep','spider','squirrel']
        self.mode = mode
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.transform = data_transform

        if self.mode == "train" or self.mode =="test":    # train mode || test mode
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.data_size += 1
                name = file.split('_')
                # label (one-hot) : [0,1,2,3,4,5,6,7,8,9]
                #                   -> {butterfly,cat,cattle,chickens,dog,elephant,horse,sheep,spider,squirrel}
                for i in range(0,10):
                    if(name[0] == self.labels[i]):
                        self.list_label.append(i)
                        break
        else:
            print('Undefined Dataset!')

    def __getitem__(self,item):
        # Train mode
        if self.mode == "train":
            img = cv2.imread(self.list_img[item])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])  # convert 'img' and 'label' to torch type
            
        # Test mode
        elif self.mode == "test":
            img = cv2.imread(self.list_img[item])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            newsize = (227,227)
            # the size of img should be (277,277)
            # the test dataset haven't been resized in data-preprocess
            img = cv2.resize(img,newsize)

            label = self.list_label[item]
            return self.transform(img),torch.LongTensor([label])
            
        else:
            print('None')
            
    def __len__(self):
        return self.data_size