from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F
from show3d_balls import showpoints
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '../pretrained_networks/latest_classification.pt',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--dataset', type=str, default="../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0", help="dataset path")
#parameters for showing critial points
parser.add_argument('--class_choice', type=str, default='Chair', help='class choice')
parser.add_argument('--idx', type=int, default=1, help='model index')
parser.add_argument('--task',type=str, default='accuracy', help='accuracy | critical points')

opt = parser.parse_args()
print(opt)
print(opt.feature_transform)
if opt.task == 'accuracy':
    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=2500)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4)


    classifier = PointNetCls(num_classes=len(test_dataset.classes),feature_transform=opt.feature_transform)
    classifier.cuda()
    classifier.load_state_dict(torch.load(opt.model)['model'])


    classifier.eval()


    total_preds = []
    total_targets = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            #TODO
            # calculate average classification accuracy
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            preds,_,_ = classifier(points,show_critical_points=False )
            pred_labels = torch.max(preds, dim=1)[1]

            total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
            total_targets = np.concatenate([total_targets, target.cpu().numpy()])
            a = 0
        accuracy = 100 * (total_targets == total_preds).sum() / len(test_dataset)
        print('Accuracy = {:.2f}%'.format(accuracy))

else:
    d = ShapeNetDataset(
        root=opt.dataset,
        class_choice=[opt.class_choice],
        split='test',
        data_augmentation=False)

    idx = opt.idx

    print("model %d/%d" % (idx, len(d)))
    point, seg = d[idx]
    print(point.size(), seg.size())
    point_np = point.numpy()


    state_dict = torch.load(opt.model)
    classifier = PointNetCls(num_classes=16,feature_transform=opt.feature_transform)
    classifier.cuda()
    classifier.load_state_dict(state_dict['model'])
    classifier.eval()

    point = point.transpose(1, 0).contiguous()

    point = Variable(point.view(1, point.size()[0], point.size()[1]))

    classifier(point.cuda(),show_critical_points = True)
    showpoints(point_np)
