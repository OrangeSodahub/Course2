"""
Caculate the accuracy using test dataset
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader as DataLoader
import time
import datetime

from model import AnimalDataSet as ADS

def test(model_dir: str, test_dir: str):
    # Load train_model
    net = torch.load(model_dir)
    # Parameter
    BATCH_SIZE = 50

    # Testset
    datafile = ADS("test", test_dir)
    dataloader = DataLoader(datafile, batch_size=BATCH_SIZE, shuffle=True)
    acc_pre = 0
    count = 0
    maxlen = 50

    start_time = time.time()
    print("[", datetime.datetime.now(), "] Test Samples: ", str(BATCH_SIZE * maxlen))
    print("[", datetime.datetime.now(), "] Test Dataset: ", test_dir)
    print("+---------------------- Begin testing --------------------------+")
    for img,label in dataloader:
        if(count == maxlen):
            break
        count += 1
        img = img.cuda()
        out = net(img)
        pred = torch.max(out, 1)[1].cpu().numpy().squeeze()
        for i in pred:
            # output the results
            # pred[i] is prediction and label[i][0] is the truth
            if(pred[i] == label[i][0].cpu().numpy()):
                acc_pre += 1

    end_time = time.time()
    print("+---------------------- End testing --------------------------+")
    print("Caculate {0} samples the accuracy is {1} time cost is {2} s".format(maxlen * BATCH_SIZE,acc_pre/(maxlen * BATCH_SIZE),(end_time - start_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="path to trained model", required=False, default="/home/zonlin/ML/Course/Course2/pkl/AlexNet_epoch200_batch200.pkl")
    parser.add_argument("--test_dir", help="path to test dataset", required=False, default="/home/zonlin/ML/Course/Course2/Animals-10/test/")
    params = parser.parse_args()
    
    if os.path.exists(params.model_dir) and os.path.exists(params.test_dir):
        test(params.model_dir, params.test_dir)
    else:
        print("No such directory: ", params.dir)