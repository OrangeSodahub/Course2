import os
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import time
from model import AlexNet
from model import AnimalDataSet as ADS

def train(dir: str):
    # Create a new Alexnet: net
    net = AlexNet(10)
    net = net.cuda()

    # Parameters
    LR = 0.001
    Epoch = 10
    BATCH_SIZE = 200
    dataset_dir = dir

    # Trainset
    datafile = ADS("train", dataset_dir) # mode, dataset_dir
    dataloader = DataLoader(datafile, batch_size=BATCH_SIZE, shuffle=True)

    # Train
    start_time = time.time()
    optimizer = torch.optim.Adam(net.parameters(),lr = LR) # Adam
    loss_func = torch.nn.CrossEntropyLoss() # CrossEntropyLoss
    cnt = 0

    print("+---------------------- Begin training ----------------------+")
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
    print("+---------------------- End training ----------------------+")

    # Save train_model
    to_dir = '/home/zonlin/ML/Course/Course2/pkl/'
    name = 'AlexNet_epoch'+str(Epoch)+'_batch'+str(BATCH_SIZE)+'.pkl'
    torch.save(net, os.path.join(to_dir, name))
    print(name, 'has been saved to ', to_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="path to train dataset", required=False, default="/home/zonlin/ML/Course/Course2/Animals-10/resize_train/")
    params = parser.parse_args()
    
    if os.path.exists(params.dir):
        train(params.dir)
    else:
        print("No such directory: ", params.dir)