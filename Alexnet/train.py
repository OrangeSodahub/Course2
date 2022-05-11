"""
对神经网络进行训练并保存
使用被resize之后的训练集resize_train
"""

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import time
from model import AlexNet
from model import AnimalDataSet as ADS

# Create a new Alexnet: net
net = AlexNet(10)
net = net.cuda()

# Parameters
LR = 0.001
Epoch = 20
BATCH_SIZE = 200
dataset_dir = r"F:/PersonalFiles/CodeWorks/Python/AlexNet/Course2/Animals-10/resize_train"

# Trainset
datafile = ADS("train",dataset_dir) # mode,dataset_dir
dataloader = DataLoader(datafile, batch_size=BATCH_SIZE, shuffle=True)

# Train
start_time = time.time()
optimizer = torch.optim.Adam(net.parameters(),lr = LR) # Adam
loss_func = torch.nn.CrossEntropyLoss() # CrossEntropyLoss
cnt = 0
for step in range(Epoch):
    for img, label in dataloader:
        img, label = Variable(img).cuda(),Variable(label).cuda()
        out = net(img)
        loss = loss_func(out,label.squeeze())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cnt += 1

        if(cnt % 5 == 0):
            print('Epoch {0}, Frame {1}, train_loss {2}'.format(step, cnt * BATCH_SIZE, loss / BATCH_SIZE))

end_time = time.time()

print("\nDone Runing Time on GPU:  %.4f"%(end_time - start_time))

torch.save(net,'net/Animal10_AlexNet.pkl')   # 保存神经网络