"""
Caculate the accuracy using test dataset
"""
import torch
from torch.utils.data import DataLoader as DataLoader
import time

from model import AlexNet
from model import AnimalDataSet as ADS

# Load train_model
net = torch.load(r"/home/zonlin/ML/Course/Course2/pkl/AlexNet_epoch10_batch200.pkl")
# Parameter
BATCH_SIZE = 50
dataset_dir = r"/home/zonlin/ML/Course/Course2/Animals-10/test/"

# Testset
datafile = ADS("test", dataset_dir)
dataloader = DataLoader(datafile, batch_size=BATCH_SIZE,shuffle=True)
acc_pre = 0
count = 0
maxlen = 50

start_time = time.time()
for img,label in dataloader:
    if(count == maxlen):
        break
    count += 1
    img = img.cuda()
    out = net(img)
    pred = torch.max(out, 1)[1].cpu().numpy().squeeze()
    for i in pred:
        # pred[i] is prediction and label[i][0] is the truth
        if(pred[i] == label[i][0].cpu().numpy()):
            acc_pre += 1

end_time = time.time()

print("Caculate {0} samples the accuracy is {1} time cost is {2} s".format(maxlen * BATCH_SIZE,acc_pre/(maxlen * BATCH_SIZE),(end_time - start_time)))