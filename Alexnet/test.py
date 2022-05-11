"""
在test测试集里选取300个图片进行测试,计算准确率
"""
import torch
from torch.utils.data import DataLoader as DataLoader
import time

from model import AlexNet
from model import AnimalDataSet as ADS

# Load train_model
net = torch.load(r"net/Animal10_AlexNet03.pkl")
# Parameter
BATCH_SIZE = 50
dataset_dir = r"F:/PersonalFiles/CourseMaterials/大二下/机器学习/Animals-10/test"

# Testset
datafile = ADS("test",dataset_dir)
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
    pred = torch.max(out, 1)[1].cpu().numpy().squeeze()  # torch.max[0]是值 [1]是index\n",
    for i in pred:
        # print(pred[i],end = ',')
        # print(label[i][0].cpu().numpy())
        if(pred[i] == label[i][0].cpu().numpy()):
            acc_pre += 1

end_time = time.time()

print("Caculate {0} samples the accuracy is {1} time cost is {2} s".format(maxlen * BATCH_SIZE,acc_pre/(maxlen * BATCH_SIZE),(end_time - start_time)))