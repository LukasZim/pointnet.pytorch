import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class FractureDataset(Dataset):
    def __init__(self, root_directory, chunk_size, dataset_type="train"):
        self.udf = {}
        self.impulse = {}
        self.dataset_type = dataset_type
        self.num_files = self._calculate_size(root_directory)
        for file in os.listdir(root_directory):
            if not file.split(".")[-1] == "pkl":
                continue
            without_filetype = file.split('.')[0]
            if len(without_filetype.split('_')) == 1:
                index = int(without_filetype)
                if self._drop_if_wrong_type(index):
                    self.udf[index] = os.path.join(root_directory, file)
            else:
                index = int(without_filetype.split('_')[0])
                if self._drop_if_wrong_type(index):
                    self.impulse[index]=os.path.join(root_directory, file)


        self.chunk_size = chunk_size
        self.data_indices = self._prepare_data_indices()

    def _drop_if_wrong_type(self, index):
        if index < self.num_files * 0.7 and self.dataset_type == "train":
            return True

        if self.num_files * 0.7 < index < self.num_files * 0.85 and self.dataset_type == "test":
            return True

        if index > self.num_files * 0.85 and self.dataset_type == "validate":
            return True

        return False

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

        asdf = pcd[start_index]
        pcd = pcd[start_index: start_index + self.chunk_size]
        udf = udf[start_index: start_index + self.chunk_size]
        label_gt = label_gt[start_index: start_index + self.chunk_size]
        label_edge = label_edge[start_index: start_index + self.chunk_size]

        pcd = torch.tensor(pcd, dtype=torch.float32)
        udf = torch.tensor(udf, dtype=torch.float32)
        impulse = torch.tensor(impulse, dtype=torch.float32)

        return pcd, udf, impulse

    def get_GT(self, idx):

        df = pd.read_pickle(self.udf[idx])
        pcd = df.drop("distance", axis=1).values
        udf = df["distance"].values


        df = pd.read_pickle(self.impulse[idx])
        impulse = df.values[0]

        return pcd, udf, impulse

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

    dataset = FractureDataset(path, chunk_size=64, dataset_type=type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, )

    return dataloader, dataset

