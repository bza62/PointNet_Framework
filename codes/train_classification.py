from __future__ import print_function
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--save_dir', default='../pretrained_networks', help='directory to save model weights')

opt = parser.parse_args()
print("opt dataset is ")
print(opt.dataset)

if not os.path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    npoints=opt.num_points)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

val_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    split='val',
    npoints=opt.num_points)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

print(len(dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

print("opt is ")
print(opt.feature_transform)
classifier = PointNetCls(num_classes=num_classes, feature_transform=opt.feature_transform)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
    print("use model")
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
classifier.cuda()

num_batch = len(dataloader)
suf = 'false'
if opt.feature_transform:
    suf = 'true'
for epoch in range(opt.nepoch):
    classifier.train()
    loss_sum = 0
    for i, data in enumerate(tqdm(dataloader, desc='Batches', leave = False), 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        #TODO
        # perform forward and backward paths, optimize network
        pred,trans, trans_feat = classifier(points,show_critical_points=False)
        loss = F.nll_loss(pred,target)
        #print(trans_feat)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans)*0.001
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_sum += loss
    print("Epoch: {} Loss is {} ".format(epoch,loss_sum/num_batch))
    if epoch%10==9:
        torch.save({'model': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}, os.path.join('/home/zbc/Assignment4/PointNet_Framework/codes/cls', str(epoch)+'_latest_classification_'+suf+'.pt'))

    torch.save({'model':classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}, os.path.join(opt.save_dir, 'latest_classification_'+suf+'.pt'))

    classifier.eval()
    total_preds = []
    total_targets = []
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            preds, _, _ = classifier(points,show_critical_points = False)
            pred_labels = torch.max(preds, dim= 1)[1]

            total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
            total_targets = np.concatenate([total_targets, target.cpu().numpy()])
            a = 0
        accuracy = 100 * (total_targets == total_preds).sum() / len(val_dataset)
        print('Accuracy = {:.2f}%'.format(accuracy))





