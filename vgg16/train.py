import torchvision.models as models
import torch
import torch.nn.functional
from torchvision import transforms
from torch import optim
from torch import nn
from sklearn.metrics import classification_report
# vgg-net
from model import Net

# Create a new vgg-net
vgg = models.vgg16(pretrained=True)
model = Net(vgg)

# Train
optimizer = optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()
 
for epoch in range(max_epoch):
    model.train()
    batch = 0
    all_loss = 0
    trail_acc = 0
    trail_totle = 0
    allax_batch = len(train_dataloader_train)
    for train_data in train_dataloader_train:
        batch_images, batch_labels = train_data
        out = model(batch_images)
        # loss
        loss = loss_func(out, batch_labels)
        all_loss += loss
        # 预测
        prediction = torch.max(out, 1)[1]
        # 总预测准确的数量
        train_correct = (prediction == batch_labels).sum()
        # 加和数量
        trail_acc += train_correct
        # 总数量
        trail_totle += len(batch_labels)
        # 求导
        optimizer.zero_grad()
        # 反向传递
        loss.backward()
        # 向前走一步
        optimizer.step()
        batch += 1
        print("Epoch: %d/%d || batch:%d/%d average_loss: %.3f || train_acc: %.2f || loss:%.2f"
              % (epoch + 1, max_epoch, batch, allax_batch, loss, train_correct / len(batch_labels), loss))
    print("Epoch: %d/%d || acc:%d || all_loss:%.2f" % (epoch + 1, max_epoch, trail_acc / trail_totle, all_loss))


# Save train_model
    transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
])
 
with torch.no_grad():
    true_lable = []
    pre_lable = []
    for train_data in train_dataloader_train:
        batch_images, batch_labels = train_data
        out = model(batch_images)
        prediction = torch.max(out, 1)[1]
        true_lable += batch_labels.tolist()
        pre_lable += prediction.tolist()
 
print(classification_report(true_lable, pre_lable))