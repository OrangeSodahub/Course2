import torch
from torch.autograd import Variable
import os
import random
import cv2
import sys

def random_read(path):
    # choose an image
    fs = os.listdir(path)
    item = random.choice(range(1, len(fs)))
    filename = path + fs[item]
    # resize the image
    newsize = (227, 227)
    img = cv2.imread(filename)
    cv2.imshow('img', img)
    # BGR TO RGB & (c,h,w) TO (h,w,c)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, newsize)
    img_resize = img_resize.transpose(2, 0, 1)
    # convert image to tensor
    t_img = torch.from_numpy(img_resize)
    t_img = Variable(torch.unsqueeze(t_img, dim=0)).type(torch.FloatTensor) / 255.    # uint8 -> {0,1}

    t_img = t_img.cuda()
    return t_img,filename

# read the predictiion
def one_hot2str(oneHot):
    labels = ['蝴蝶','猫','牛','鸡','狗','大象','马','羊','蜘蛛','松鼠']
    pred = torch.max(oneHot,1)[1].cpu().numpy().squeeze()    # torch.max[0]是值 [1]是index
    str1 = labels[pred]
    str1 = "Predict: " + str1
    return str1

def main(argv):
    img,filename = random_read(argv)
    # Load train_model
    net = torch.load(r"/home/zonlin/ML/Course/Course2/pkl/AlexNet_epoch10_batch200.pkl")
    pred_hat = net(img)
    pred_str = one_hot2str(pred_hat)
    return pred_str, filename

if __name__ == "__main__":
    pred_str, filename = main(sys.argv[1])
    print(pred_str)
    print(filename)