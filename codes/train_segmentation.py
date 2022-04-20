from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--save_dir', default='../pretrained_networks', help='directory to save model weights')

opt = parser.parse_args()
print(opt)
print(opt.feature_transform)
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

val_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(val_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = PointNetDenseCls(num_classes=num_classes, feature_transform=opt.feature_transform)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
classifier.cuda()

classifier_loss = nn.NLLLoss()
num_batch = len(dataloader)
suf = 'false'
if opt.feature_transform:
    suf = 'true'
for epoch in range(opt.nepoch):
    classifier.train()
    loss_sum = 0
    total_correct = 0
    for i, data in enumerate(tqdm(dataloader, desc='Batches', leave = False), 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        #TODO
        # perform forward and backward paths, optimize network
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1,num_classes)
        target = target.view(-1,1)[:,0]-1
        #print(trans_feat)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            # print("trans_feat is ")
            # print(trans_feat)
            loss += feature_transform_regularizer(trans) * 0.001
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred_choice = pred.data.max(1)[1]
        total_correct += pred_choice.eq(target.data).cpu().sum()
        loss_sum += loss
    print("Epoch: {} Loss is {} accuarancy is {}".format(epoch, loss_sum / num_batch, total_correct.item() / float(opt.batchSize * num_batch * 2500)))

    if epoch % 10 == 9:
        torch.save({'model': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch},
                   os.path.join('/home/zbc/Assignment4/PointNet_Framework/codes/seg', str(epoch)+'_latest_segmentation_'+suf+'.pt'))

    torch.save({'model':classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}, os.path.join(opt.save_dir, 'latest_segmentation_'+suf+'.pt'))


## benchmark mIOU
    classifier.eval()
    shape_ious = []
    with torch.no_grad():
        for i,data in tqdm(enumerate(val_dataloader, 0)):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(2)[1]

            pred_np = pred_choice.cpu().data.numpy()
            target_np = target.cpu().data.numpy() - 1

            for shape_idx in range(target_np.shape[0]):
                parts = range(num_classes)#np.unique(target_np[shape_idx])
                part_ious = []
                for part in parts:
                    I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                    if U == 0:
                        iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                    else:
                        iou = I / float(U)
                    part_ious.append(iou)
                shape_ious.append(np.mean(part_ious))

        print("mIOU for class {}: {:.4f}".format(opt.class_choice, np.mean(shape_ious)))
