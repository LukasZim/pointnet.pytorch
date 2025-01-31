import random

import pandas as pd
from torch.utils.data import Dataset, DataLoader


class FractureDataset(Dataset):
    def __init__(self, root_directory, chunk_size, dataset_type="train"):
        self.udf = {}
        self.impulse = {}
        self.dataset_type = dataset_type
        self.num_files_directory = self._calculate_size(root_directory)
        self.files_used = 0
        for file in os.listdir(root_directory):
            if not file.split(".")[-1] == "pkl":
                continue
            without_filetype = file.split('.')[0]
            if len(without_filetype.split('_')) == 1:
                index = int(without_filetype)
                correct, index = self._drop_if_wrong_type(index)
                if correct:
                    self.udf[index] = os.path.join(root_directory, file)
                    self.files_used += 1
            else:
                index = int(without_filetype.split('_')[0])
                correct, index = self._drop_if_wrong_type(index)
                if correct:
                    self.impulse[index] = os.path.join(root_directory, file)

        self.chunk_size = chunk_size
        self.data_indices = self._prepare_data_indices()

    def _drop_if_wrong_type(self, index):
        if index <= self.num_files_directory * 0.7 and self.dataset_type == "train":
            return True, index

        if self.num_files_directory * 0.7 < index < self.num_files_directory * 0.85 and self.dataset_type == "test":
            return True, index - int(self.num_files_directory * 0.7) - 1

        if index > self.num_files_directory * 0.85 and self.dataset_type == "validate":
            return True, index - int(self.num_files_directory * 0.85) - 1

        return False, -1

    def _calculate_size(self, directory):
        size = 0
        for file in os.listdir(directory):
            if not file.split(".")[-1] == "pkl":
                continue
            without_filetype = file.split('.')[0]
            if len(without_filetype.split('_')) == 1:
                index = int(without_filetype)
            else:
                index = int(without_filetype.split('_')[0])

            if index > size:
                size = index

        return size + 1

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        file_index, start_index = self.data_indices[idx]

        filename = self.udf[file_index]
        df = pd.read_pickle(filename)
        pcd = df.drop(["distance", "label", "edge_labels"], axis=1).values
        udf = df["distance"].values
        label_gt = df["label"].values
        label_edge = df['edge_labels'].values

        df = pd.read_pickle(self.impulse[file_index])
        impulse = df.values[0]

        pcd = pcd[start_index: start_index + self.chunk_size]
        udf = udf[start_index: start_index + self.chunk_size]
        label_gt = label_gt[start_index: start_index + self.chunk_size]
        label_edge = label_edge[start_index: start_index + self.chunk_size]

        pcd = torch.tensor(pcd, dtype=torch.float32)
        udf = torch.tensor(udf, dtype=torch.float32)
        impulse = torch.tensor(impulse, dtype=torch.float32)

        return pcd, udf, impulse, label_gt, label_edge

    def get_GT(self, idx):

        df = pd.read_pickle(self.udf[idx])
        pcd = df.drop(["distance", "label", "edge_labels"], axis=1).values
        udf = df["distance"].values
        label_gt = df["label"].values
        label_edge = df['edge_labels'].values

        df = pd.read_pickle(self.impulse[idx])
        impulse = df.values[0]

        return pcd, udf, impulse, label_gt, label_edge

    def get_random_GT(self):
        index = random.randint(0, self.files_used - 1)
        return self.get_GT(index)

    def _prepare_data_indices(self):
        indices = []
        for file_index, path in self.udf.items():

            num_points = pd.read_pickle(path).shape[0]
            for start_index in range(0, num_points - self.chunk_size, self.chunk_size):
                indices.append((file_index, start_index))

        return indices


def variable_size_collate_fn(batch):
    # Batch is a list of tensors
    print(type(batch))
    print(len(batch))
    return batch


def FractureDataLoader(path, type):
    # df = pd.read_pickle(path)
    # print(df.head())

    # X = df.drop("distance", axis=1).values
    # y = df["distance"].values
    # # y[y > 0.3] = 0.3
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dataset = FractureDataset(path, chunk_size=3300, dataset_type=type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, )

    return dataloader, dataset


import os
import os.path as osp
import shutil
import json

import torch

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array


class ShapeNet(InMemoryDataset):

    def __init__(self, root, include_normals=True,
                 split='trainval', transform=None, pre_transform=None,
                 pre_filter=None):

        super(ShapeNet, self).__init__(root, transform, pre_transform,
                                       pre_filter)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        elif split == 'trainval':
            path = self.processed_paths[3]
        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))

        self.data, self.slices = torch.load(path)
        self.data.x = self.data.x if include_normals else None

        self.y_mask = torch.zeros((len(self.seg_classes.keys()), 50),
                                  dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            self.y_mask[i, labels] = 1

    @property
    def num_classes(self):
        return self.y_mask.size(-1)



    def process_filenames(self, filenames):
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}
        if self.n_per_class is not None:
            per_class_count = [self.n_per_class for i in range(len(cat_idx))]

        for i, name in enumerate(filenames):
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue
            if self.n_per_class is not None:
                if per_class_count[cat_idx[cat]] <= 0:
                    continue
                per_class_count[cat_idx[cat]] -= 1

            data = read_txt_array(osp.join(self.raw_dir, name))
            pos = data[:, :3]
            x = data[:, 3:6]
            y = data[:, -1].type(torch.long)
            cat_onehot = pos.new_zeros(1, 16)
            cat_onehot[0, cat_idx[cat]] = 1
            data = Data(pos=pos, norm=x, y=y, category=cat_onehot)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        trainval = []
        for i, split in enumerate(['train', 'val', 'test']):
            path = osp.join(self.raw_dir, 'train_test_split',
                            f'shuffled_{split}_file_list.json')
            with open(path, 'r') as f:
                filenames = [
                    osp.sep.join(name.split('/')[1:]) + '.txt'
                    for name in json.load(f)
                ]  # Removing first directory.
            data_list = self.process_filenames(filenames)
            if split == 'train' or split == 'val':
                trainval += data_list
            torch.save(self.collate(data_list), self.processed_paths[i])
        torch.save(self.collate(trainval), self.processed_paths[3])

    def __repr__(self):
        return '{}({}, categories={})'.format(self.__class__.__name__,
                                              len(self), self.categories)