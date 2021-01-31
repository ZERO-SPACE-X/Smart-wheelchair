import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import visdom
import tqdm
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import DataLoader
from option import BasciOption
from models import create_dataloader, Classification,CnnLstm, CnnLstmPlus
import numpy as np
def train_one_epoch(opt):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataloader = create_dataloader(opt)
    net = CnnLstmPlus()  # 定义训练的网络模型
    net.to(device)
    net.train()
    # loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
    loss_function = nn.BCELoss()  # 定义损失函数为交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 定义优化器（训练参数，学习率）

    # for epoch in range(opt.num_epochs):  # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0
    correct = 0
    total = 0

    time_start = time.perf_counter()

    for step, data in enumerate(train_dataloader,
                                start=0):  # 遍历训练集，step从0开始计算
    # for data in tqdm(train_dataloader):
        inputs, labels = data # 获取训练集的图像和标签
        inputs, labels = inputs.to(device), labels.to(device)
        # print(f'labels before.{labels}')
        optimizer.zero_grad()  # 清除历史梯度

        # forward + backward + optimize
        # outputs = net(inputs.permute(0,1,3,2))  # 正向传播
        outputs = net(inputs)  # 正向传播
        # print(f'debug outputs.shape,label.shape:{outputs.size(),labels.size()}')
        # outputs = torch.max(outputs, dim=1)[1]
        loss = loss_function(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数
        # predict_y = torch.max(outputs, dim=1)[1]
        # outputs = outputs.numpy()
        # print(f'outputs:{outputs}\n,labels:{labels}')
        zero = torch.zeros_like(outputs)
        one = torch.ones_like(outputs)
        outputs = torch.where(outputs > 0.5, one, outputs)
        outputs = torch.where(outputs < 0.5, zero, outputs)

        # if (outputs - 0.5) > 0:
        #     outputs = 1
        # else:
        #     outputs = 0
        # print('debug', outputs.size())
        # print(f'outputs:{outputs}\n,labels:{labels}')
        total += labels.size(0)
        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        if outputs.all() == labels.all():
            correct += 1
        # correct += (predict_y.item() == labels).sum()
        # correct += (predict_y.item() == labels)
        running_loss += loss.item()
        # print statistics

        # print('train_dataloader length: ', len(train_dataloader))
        # print(f'batch loss : {running_loss}')
    losses = running_loss / total
    acc = correct / total
    # print('Train on epoch {}: loss:{}, acc:{}%'.format(epoch + 1, running_loss / total, 100 * correct / total))
    print(f'Train on one epoch: loss:{losses}, acc:{100 * acc}%')
    # 保存训练得到的参数
    if opt.model == 'basic':
        save_weight_name = os.path.join(opt.save_path,
                                        'Basic_Epoch_{0}_Accuracy_{1:.2f}.pth'.format(
                                            '1',
                                            acc))
    elif opt.model == 'plus':
        save_weight_name = os.path.join(opt.save_path,
                                        'Plus_Epoch_{0}_Accuracy_{1:.2f}.pth'.format(
                                            '1',
                                            acc))
    torch.save(net.state_dict(), save_weight_name)
    # print('Finished Training')
    return losses, acc


if __name__ == '__main__':
    is_train = True
    opt = BasciOption()
    args = opt.initialize(is_train)
    opt.print_args(args)
    viz = visdom.Visdom(env='CnnLstmPlus-fc1-1024-lstm-512-2')

    viz.line([[0.,0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'acc']))
    for i in range(args.num_epochs):
        print(f'--------------------epoch:{i}--------------------')
        losses, acc = train_one_epoch(args)
        # 随机获取loss和acc
        # loss = 0.1 * np.random.randn() + 1
        # acc = 0.1 * np.random.randn() + 0.5
        # 更新窗口图像
        viz.line([[losses, acc]], [i], win='train', update='append')
        # 延时0.5s
        time.sleep(0.5)


