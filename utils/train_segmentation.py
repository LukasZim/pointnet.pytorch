from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, SplatDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer, SplatNetDenseCls
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# parsing arguments
parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--batchSize', type=int, default=16, help='input batch size')
# parser.add_argument(
#     '--workers', type=int, help='number of data loading workers', default=4)
# parser.add_argument(
#     '--nepoch', type=int, default=1000, help='number of epochs to train for')
# parser.add_argument('--outf', type=str, default='seg', help='output folder')
# parser.add_argument('--model', type=str, default='', help='model path')
# parser.add_argument('--dataset', type=str, required=True, help="dataset path")
# parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
# parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
# parser.add_argument('--dataset_type', type=str, default='splat', help="dataset type splat|shapenet|modelnet40")
#
opt = parser.parse_args()
print(opt)

opt.batchSize = 16
opt.workers = 8
opt.nepoch = 1000
opt.outf = "seg"
opt.model = ""
opt.dataset = "/home/lukasz/Documents/thesis_pointcloud/dataset/Chair"
opt.class_choice = "Chair"
opt.dataset_type = "splat"
opt.feature_transform = False

# setting seed
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == "splat":
    dataset = SplatDataset(opt.dataset,
                           split='train')
    dataloader = torch.utils.data.DataLoader(dataset,
                     batch_size=opt.batchSize,
                     shuffle=True,
                     num_workers=int(opt.workers))

    test_dataset = SplatDataset(opt.dataset,
                           split='test')
    testdataloader = torch.utils.data.DataLoader(test_dataset,
                     batch_size=opt.batchSize,
                     shuffle=True,
                     num_workers=int(opt.workers))

else:
# retrieving dataset and dataloaders
#     raise Exception("Not used in current build")
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        class_choice=[opt.class_choice])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        class_choice=[opt.class_choice],
        split='test',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

# setting up directories
print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass
blue = lambda x: '\033[94m' + x + '\033[0m'

# setting up the model
classifier = SplatNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
print(torch.cuda.is_available())
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

# starting the training
for epoch in range(opt.nepoch):

    # going over the dataset
    for i, data in enumerate(dataloader, 0):
        points, target, impulses = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]

        _,class_counts = torch.unique(target, return_counts=True)
        class_counts = class_counts.tolist()
        total_samples = sum(class_counts)
        class_weights = [total_samples / class_count for class_count in class_counts]
        class_weights.extend([100000.0] * (num_classes - len(class_weights)))
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).cuda()

        # Assuming 'num_classes' is the number of output classes in your model
        assert torch.min(target) >= 0, f"Target contains negative values: {torch.min(target)}"
        assert torch.max(target) < num_classes, f"Target contains values out of range: {torch.max(target)}"

        loss = F.nll_loss(pred, target, weight=weights_tensor)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, impulse = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))
    scheduler.step()
    print(opt.outf, opt.class_choice, epoch)
    print('%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
    if epoch % 25 == 0:
        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target, impulse = data
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