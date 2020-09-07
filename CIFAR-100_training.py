#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:13:27 2020

@author: lee
"""


import torch
from torchvision import datasets
import torch.nn as nn
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wresnet import *
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

class DataParallelCriterion(DataParallel):
    def forward(self, inputs, *targets, **kwargs):
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        targets = tuple(targets_per_gpu[0] for targets_per_gpu in targets)
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs), targets
    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.shape) > 1:
        target = torch.argmax(target, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
   
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        activation1 = out
        out = self.block2(out)
        activation2 = out
        out = self.block3(out)
        activation3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), activation1, activation2, activation3
    
    
def main(args):
    transform = transforms.Compose([transforms.ColorJitter(), 
                                    transforms.RandomResizedCrop(32, (0.5, 1.5)), 
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071,0.4867, 0.4408), (0.2675,0.2565,0.2761))])
    
    train_dataset = datasets.CIFAR100('./Datasets', train=True, transform=transform, download=True)
    test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071,0.4867, 0.4408), (0.2675,0.2565,0.2761))])
    test_dataset = datasets.CIFAR100('./Datasets', train=False, transform=test_transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    net_depth, net_width = int(args.select_model[4:6]), int(args.select_model[-1])
    Teacher_model = WideResNet(depth=net_depth, num_classes=100, widen_factor=net_width, dropRate=0.3)
    # Teacher_model = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    Teacher_model.cuda()
    Teacher_model = torch.nn.DataParallel(Teacher_model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    optimizer = torch.optim.SGD(Teacher_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(args.total_epochs):
        
        for iter_, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to('cuda:0'), labels.to('cuda:0')
            outputs, *acts = Teacher_model(images)
            classification_loss = criterion(outputs, labels)
            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()
            if iter_%10==0:
                print("Epoch: {}/{}, Iteration: {}/{}, Loss: {:02.5f}".format(epoch,args.total_epochs, iter_, train_loader.__len__(), classification_loss.item()))
            
        with torch.no_grad():
            Teacher_model.eval()
            cumulated_acc=0
            for x, y in test_loader:
                x, y = x.to('cuda:0'), y.to('cuda:0')
                logits, *activations = Teacher_model(x)
                acc = accuracy(logits.data, y, topk=(1,))[0]
                cumulated_acc+=acc
            print("Test Accuracy is {:02.2f} %".format(cumulated_acc/test_loader.__len__()*100))
            Teacher_model.train()
            if best_acc <= cumulated_acc/test_loader.__len__()*100:
               best_acc = cumulated_acc/test_loader.__len__()*100
               torch.save(Teacher_model.state_dict(),'./Pretrained/CIFAR100/WRN-{}-{}/Teacher_best.ckpt'.format(net_depth, net_width))
        
        optim_scheduler.step()

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='')
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--batch_size', type=int, default=128, help='total training batch size')
    parser.add_argument('--select_model', type=str, default='WRN-40-2', help='What do you want to train?')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    main(args)
