from __future__ import print_function

import copy
from show3d_balls import showpoints
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
#from show3d_balls import showpoints

class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()

        self.k = k

        # Each layer has batchnorm and relu on it
        #TODO
        # layer 1: k -> 64
        #TODO
        # layer 2:  64 -> 128
        #TODO
        # layer 3: 128 -> 1024
        self.layers = nn.Sequential(
            nn.Conv1d(k,64,kernel_size=1), nn.BatchNorm1d(64),nn.ReLU(True),
            nn.Conv1d(64,128,kernel_size=1),nn.BatchNorm1d(128),nn.ReLU(True),
            nn.Conv1d(128,1024,kernel_size=1),nn.BatchNorm1d(1024),nn.ReLU(True))
        #TODO
        # fc 1024 -> 512
        # #TODO
        # fc 512 -> 256
        #TODO
        # fc 256 -> k*k (no batchnorm, no relu)
        self.fc = nn.Sequential(nn.Linear(1024,512),nn.Linear(512,256),nn.Linear(256,k*k))
        #TODO
        # ReLU activationfunction


    def forward(self, x):
        batch_size, _, num_points = x.shape
        #TODO
        # apply layer 1
        #TODO
        # apply layer 2
        #TODO
        # apply layer 3
        x = self.layers(x)
        #TODO
        # do maxpooling and flatten
        x = torch.max(x, dim=2)[0]
        #TODO
        # apply fc layer 1
        #TODO
        # apply fc layer
        #TODO
        # apply fc layer 3
        #TODO
        #reshape output to a b*k*k tensor
        x = self.fc(x).view(-1, self.k, self.k)
        #TODO
        # define an identity matrix to add to the output. This will help with the stability of the results since we want our transformations to be close to identity
        bias = torch.eye(self.k, requires_grad=True).repeat(batch_size,1,1)
        if x.is_cuda:
            bias = bias.cuda()

        #TODO
        # return output
        return  x+bias


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = True):
        super(PointNetfeat, self).__init__()

        self.feature_transform= feature_transform
        self.global_feat = global_feat

        #TODO
        # Use TNet to apply transformation on input and multiply the input points with the transformation
        self.t_net3 = TNet(k= 3)

        #TODO
        # layer 1:3 -> 64
        self.layer1 = nn.Sequential(nn.Conv1d(3,64,1),nn.BatchNorm1d(64),nn.ReLU(True))
        #TODO
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)
        if feature_transform:
            self.t_net64 = TNet(k=64)
        #TODO
        # layer2: 64 -> 128
        #TODO
        # layer 3: 128 -> 1024 (no relu)
        self.layer2_3 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))
        #TODO
        # ReLU activation
    def forward(self, x, show_critical_points = False):
        batch_size, _, num_points = x.shape
        #TODO
        # input transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        input_trans = self.t_net3(x)
        x_original = x
        x = torch.bmm(torch.transpose(x,1,2),input_trans)
        x = torch.transpose(x,1,2)
        #TODO
        # apply layer 1
        x = self.layer1(x)
        #TODO
        # feature transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        feature_trans = None

        # print("feature_transform is ")
        # print(self.feature_transform)
        if self.feature_transform:

            feature_trans = self.t_net64(x)
            # print("feature_trans is ")
            # print(feature_trans)
            x = torch.bmm(torch.transpose(x, 1, 2), feature_trans)
            x = torch.transpose(x, 1, 2)

        point_features = x
        #TODO
        # apply layer 2
        #TODO
        # apply layer 3zbc89631412
        x = self.layer2_3(x)
        #TODO
        # apply maxpooling
        x,idx = torch.max(x, dim=2)
        if show_critical_points:
            critical_points = x_original.permute(0,2,1)[0,idx]
            showpoints(critical_points[0].cpu().numpy())
        #TODO
        # return output, input transformation matrix, feature transformation matrix
        if self.global_feat: # This shows if we're doing classification or segmentation
            return x, input_trans, feature_trans

        else:
            x = x.view(-1,1024,1).repeat(1,1,num_points)
            return torch.cat([x,point_features],1),input_trans,feature_trans



class PointNetCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, show_critical_points):
        x, trans, trans_feat = self.feat(x,show_critical_points)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        #TODO
        # get global features + point features from PointNetfeat
        self.k = num_classes
        self.feat = PointNetfeat(global_feat=False,feature_transform=feature_transform)
        #TODO
        # layer 1: 1088 -> 512
        #TODO
        # layer 2: 512 -> 256
        #TODO
        # layer 3: 256 -> 128
        #TODO
        # layer 4:  128 -> k (no ru and batch norm)
        #TODO
        # ReLU activation
        self.layers =  nn.Sequential(nn.Conv1d(1088,512,1),nn.BatchNorm1d(512),nn.ReLU(True),
                                     nn.Conv1d(512,256,1),nn.BatchNorm1d(256),nn.ReLU(True),
                                     nn.Conv1d(256,128,1),nn.BatchNorm1d(128),nn.ReLU(True),
                                     nn.Conv1d(128,num_classes,1))
    def forward(self, x):
        #TODO
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)
        b = x.size(0)
        n = x.size(2)
        x, trans, trans_feat = self.feat(x)# b*(1024+64)*n
        #TODO
        # apply layer 1
        #TODO
        # apply layer 2
        #TODO
        # apply layer 3
        #TODO
        # apply layer 4
        x = self.layers(x) #b*k*n
        #TODO
        # apply log-softmax
        x = x.transpose(2,1).contiguous() #b*n*k
        x = F.log_softmax(x.view(-1,self.k),dim=-1)# (b*N)*k
        x = x.view(b,n,self.k) # b*n*k
        return x, trans, trans_feat


def feature_transform_regularizer(trans):

    batch_size, feature_size, _ = trans.shape
    #TODO
    # compute I - AA^t
    I = torch.unsqueeze(torch.eye(feature_size),0)
    I = Variable(I)
    #TODO
    # compute norm
    if trans.is_cuda:
        I = I.cuda()
    I_AAT = I - torch.bmm(trans,torch.transpose(trans,1,2))
    norm = torch.norm(I_AAT,dim = (1,2))
    #TODO
    # compute mean norms and return
    return torch.mean(norm)



if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(num_classes = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(num_classes = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())

