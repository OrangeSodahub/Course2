import argparse
import torch
from torch.autograd import Variable
import os
import random
import cv2
import sys

def process(dir: str):
    # resize the image
    newsize = (227, 227)
    img = cv2.imread(dir)
    # BGR TO RGB & (c,h,w) TO (h,w,c)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, newsize)
    img_resize = img_resize.transpose(2, 0, 1)
    # convert image to tensor
    t_img = torch.from_numpy(img_resize)
    t_img = Variable(torch.unsqueeze(t_img, dim=0)).type(torch.FloatTensor) / 255.    # uint8 -> {0,1}

    t_img = t_img.cuda()
    return t_img

# read the predictiion
def one_hot2str(oneHot):
    labels = ['butterfly','cat','cow','chicken','dog','elephant','horse','sheep','spider','squirrel']
    pred = torch.max(oneHot,1)[1].cpu().numpy().squeeze()
    return labels[pred]

def main(dir: str):
    img = process(dir)
    # Load train_model
    net = torch.load(r"/home/zonlin/ML/Course/Course2/pkl/AlexNet_epoch200_batch200.pkl")
    pred_hat = net(img)
    pred_str = one_hot2str(pred_hat)
    return pred_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help="path to img", required=False, default="/home/zonlin/ML/Course/Course2/pred/0000.jpg")
    params = parser.parse_args()
    
    if os.path.exists(params.img_dir):
        pred_str = main(params.img_dir)
        print("Prediction: ", pred_str)
        print("Image path: ", params.img_dir)
    else:
        print("No such file: ", params.img_dir)