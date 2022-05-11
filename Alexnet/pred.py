import torch
from torch.autograd import Variable
import os
import random
import cv2
import sys

from model import AlexNet

# 随机选取图片来读 并修改为可以读取的格式
def random_read(path):
    # 随机选图片  从fs文件列表里读取一个名字
    fs = os.listdir(path)
    item = random.choice(range(1, len(fs)))
    filename = path + fs[item]    # 因为java端已经设置过了末尾加/  这里就不需要加了
    # 图片resize为227*227
    newsize = (227, 227)
    img = cv2.imread(filename)
    # BGR TO RGB & (c,h,w) TO (h,w,c)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, newsize)
    img_resize = img_resize.transpose(2, 0, 1)
    t_img = torch.from_numpy(img_resize)     # 改为tensor
    t_img = Variable(torch.unsqueeze(t_img, dim=0)).type(torch.FloatTensor) / 255.    # uint8 -> {0,1}

    t_img = t_img.cuda()
    return t_img,filename

# 解析预测结果
def one_hot2str(oneHot):
    labels = ['蝴蝶','猫','牛','鸡','狗','大象','马','羊','蜘蛛','松鼠']
    pred = torch.max(oneHot,1)[1].cpu().numpy().squeeze()    # torch.max[0]是值 [1]是index
    str1 = labels[pred]
    str1 = "预测为：" + str1
    return str1

# 执行函数
def mfunc(argv):
    img,filename = random_read(argv)
    # Load train_model
    net = torch.load(r"net/Animal10_AlexNet02.pkl")
    pred_hat = net(img)
    pred_str = one_hot2str(pred_hat)
    return pred_str,filename

if __name__ == "__main__":
    # 预测检验
    pred_str,filename = mfunc(sys.argv[1])
    print(pred_str)
    print(filename)