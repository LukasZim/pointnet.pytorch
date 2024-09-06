from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--dataset_type', type=str, default='splat', help="dataset type splat|shapenet|modelnet40")

opt = parser.parse_args()
print(opt)

# setting seed
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
pcd_list = []
pcd_list_test = []
labels_list = []
labels_list_test = []
impulse_info_list = []
impulse_info_test = []
if opt.dataset_type == "splat":
    for file in os.listdir(opt.dataset + "/points"):
        filename = os.fsdecode(file)
        pcd = []
        for line in open(opt.dataset + "/points/" + filename):
            pcd.append(list(map(lambda x: float(x), line.split(" "))))
        pcd_list.append(pcd)

    for file in os.listdir(opt.dataset + "/points_label"):
        filename = os.fsdecode(file)
        labels = []
        for line in open(opt.dataset + "/points_label/" + filename):
            labels.append(int(line))
        labels_list.append(labels)

    for file in os.listdir(opt.dataset + "/impulse_info"):
        filename = os.fsdecode(file)
        impulse_info = None
        for line in open(opt.dataset + "/impulse_info/" + filename):
            impulse_info = list(map(lambda x: float(x), line.split(" ")))
        impulse_info_list.append(impulse_info)

    eighty_percent_index = int(len(pcd_list) * 0.8)
    pcd_list_test = pcd_list[eighty_percent_index:]
    pcd_list = pcd_list[:eighty_percent_index]

    labels_list_test = labels_list[eighty_percent_index:]
    labels_list = labels_list[:eighty_percent_index]

    impulse_info_test = impulse_info_list[eighty_percent_index:]
    impulse_info_list = impulse_info_list[:eighty_percent_index]

else:
# retrieving dataset and dataloaders
    raise Exception("Not used in current build")
#     dataset = ShapeNetDataset(
#         root=opt.dataset,
#         classification=False,
#         class_choice=[opt.class_choice])
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=opt.batchSize,
#         shuffle=True,
#         num_workers=int(opt.workers))
#
#     test_dataset = ShapeNetDataset(
#         root=opt.dataset,
#         classification=False,
#         class_choice=[opt.class_choice],
#         split='test',
#         data_augmentation=False)
#     testdataloader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=opt.batchSize,
#         shuffle=True,
#         num_workers=int(opt.workers))

# setting up directories
print(len(pcd_list), len(pcd_list_test))
num_classes = np.max(np.array(labels_list))
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass
blue = lambda x: '\033[94m' + x + '\033[0m'

# setting up the model
classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(pcd_list) / opt.batchSize

# starting the training
for epoch in range(opt.nepoch):

    # going over the dataset
    for i in enumerate(len(pcd_list), 0):
        points = pcd_list[i]
        target = labels_list[i]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))
    scheduler.step()
    print(opt.outf, opt.class_choice, epoch)
    print('%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
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

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))