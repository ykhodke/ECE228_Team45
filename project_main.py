#!/usr/bin/env python
# coding: utf-8

# # References
#    1. Dataset
#        <a href="https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#225166291b3530e3e3034786a37019aba53cafba">The Cancer Imaging Archive (TCIA) Public Access</a>
#         
#    2. Pydicom
#        <a href="https://pydicom.github.io/pydicom/stable/">Pydicom</a>
#  
#         
# 

##### Importing all the neccesary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.io import read_image
import torchvision.models as models

import os
import re, glob, shutil
import time

#import pydicom
from PIL import Image


from models.model_cnn import *


# #### Defining class needed for using custom dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #ptransform = transforms.ToTensor()
        #image = ptransform(Image.open(img_path))
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# #### Creating train and test data loaders

train_dataset_directory = './dataset/train/'
train_csv = './dataset/train.csv'

test_dataset_directory = './dataset/test/'
test_csv = './dataset/test.csv'

label_dict = {
  0 : "MALIGNANT",
  1 : "BENIGN",
  2 : "BENIGN_WTIHOUT_CALLBACK"
}

training_data = CustomImageDataset(
    annotations_file = train_csv,
    img_dir = train_dataset_directory,
    transform = transforms.Compose([
        transforms.RandomCrop(256, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)))
    ]))

testing_data = CustomImageDataset(
    annotations_file = test_csv,
    img_dir = test_dataset_directory,
    transform = transforms.Compose([
        transforms.RandomCrop(256, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]))

batch_size = 16

train_dataloader = DataLoader(training_data, batch_size= batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size= batch_size, shuffle=True)


# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# print("this is the image dtype {}".format(img.dtype))
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label} ({label_dict[label.item()]})")


# #### Defining the model

model_inuse = 'cnn_test'
model = CNN_test(model_inuse)
print(model)


print_freq = 16 # every 100 batches, accuracy printed. Here, each batch includes "batch_size" data points
# CIFAR10 has 50,000 training data, and 10,000 validation data.

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [80, 120]
    if epoch in adjust_list: 
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            
            
def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    loss_to_plot = 0
    prec_to_plot = 0
    for i, (input, target) in enumerate(trainloader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        input = input.float()
        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        loss_to_plot += loss.item()
        prec_to_plot += prec.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    loss_to_plot /= 83
    prec_to_plot /= 83

    return loss_to_plot, prec_to_plot

def validate(val_loader, model, criterion ):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            
            input = input.float()
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:  # This line shows how frequently print out the status. e.g., i%5 => every 5 batch, prints out
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))
    return top1.avg


loss_yk = np.zeros([1])
train_acc_yk = np.zeros([1])
valid_acc_yk = np.zeros([1])

lr = 1e-3
weight_decay = 1e-4
epochs = 250
best_prec = 0

model = nn.DataParallel(model).cuda()
model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr =lr, momentum=0.9, weight_decay=weight_decay)
#cudnn.benchmark = True

if not os.path.exists('result'):
    os.makedirs('result')
fdir = 'result/'+str(model_inuse)
if not os.path.exists(fdir):
    os.makedirs(fdir)
        

for epoch in range(0, epochs):
    adjust_learning_rate(optimizer, epoch)

    ykl, ykp = train(train_dataloader, model, criterion, optimizer, epoch)
    
    # evaluate on test set
    print("Validation starts")
    prec = validate(test_dataloader, model, criterion)

    if(epoch == 0):
        loss_yk = np.array(ykl)
        train_acc_yk = np.array(ykp)
        valid_acc_yk = np.array(prec)
    else:
        loss_yk = np.append(loss_yk, ykl)
        train_acc_yk = np.append(train_acc_yk, ykp)
        valid_acc_yk = np.append(valid_acc_yk, prec)


loss_yk = np.array(loss_yk)
train_acc_yk = np.array(train_acc_yk)
valid_acc_yk = np.array(valid_acc_yk)

epochs_plt = np.arange(0, epochs)

plt.title("Model Train Loss over Time") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.plot(epochs_plt, loss_yk, label = "training_loss")
plt.legend()
plt.show()
plt.savefig('train_loss.png')

plt.title("Model Train Accuracy over Time") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.plot(epochs_plt, train_acc_yk, label = "training_accuracy")
plt.legend()
plt.show()
plt.savefig('train_accuracy.png')

plt.title("Model Testing Accuracy over Time") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.plot(epochs_plt, valid_acc_yk, label = "training_accuracy")
plt.legend()
plt.show()
plt.savefig('train_accuracy.png')








