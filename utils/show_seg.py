from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset, SplatDataset
from pointnet.model import PointNetDenseCls, SplatNetDenseCls
import matplotlib.pyplot as plt
import time
import cProfile


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()
#
# parser.add_argument('--model', type=str, default='', help='model path')
# parser.add_argument('--idx', type=int, default=3, help='model index')
# parser.add_argument('--dataset', type=str, default='', help='dataset path')
# parser.add_argument('--class_choice', type=str, default='', help='class choice')

opt = parser.parse_args()
opt.model = "/home/lukasz/Documents/pointnet.pytorch/utils/seg/seg_model_Chair_975.pth"
opt.dataset = "/home/lukasz/Documents/thesis_pointcloud/dataset/chair"
opt.idx = 1
opt.class_choice = ""
print(opt)
idx = opt.idx

# d = ShapeNetDataset(
#     root=opt.dataset,
#     class_choice=[opt.class_choice],
#     split='test',
#     data_augmentation=False)
start_time = time.perf_counter()
# cProfile.run("SplatDataset(path=opt.dataset, split=\"test\", data_augmentation=False)")
d = SplatDataset(path=opt.dataset, split="test", data_augmentation=False)
print("loading dataset duration: " , time.perf_counter() - start_time)


start_time = time.perf_counter()
print("model %d/%d" % (idx, len(d)))
point, seg, impulse = d[idx]
print(point.size(), seg.size())
point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]
print("other steps duration: " , time.perf_counter() - start_time)


start_time = time.perf_counter()
state_dict = torch.load(opt.model)
classifier = SplatNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval()
print("loading model duration: " , time.perf_counter() - start_time)

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0] - 1, :]

#print(pred_color.shape)
showpoints(point_np[:,:3], gt, pred_color)
