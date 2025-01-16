import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class FractureDataset(Dataset):
    def __init__(self, root_directory, chunk_size):
        self.udf = {}
        self.impulse = {}
        for file in os.listdir(root_directory):
            if not file.split(".")[-1] == "pkl":
                continue
            without_filetype = file.split('.')[0]
            if len(without_filetype.split('_')) == 1:
                index = int(without_filetype)
                self.udf[index] = os.path.join(root_directory, file)
            else:
                index = int(without_filetype.split('_')[0])
                self.impulse[index]=os.path.join(root_directory, file)


        self.chunk_size = chunk_size
        self.data_indices = self._prepare_data_indices()

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        file_index, start_index = self.data_indices[idx]


        filename = self.udf[file_index]
        df = pd.read_pickle(filename)
        pcd = df.drop("distance", axis=1).values
        udf = df["distance"].values

        df = pd.read_pickle(self.impulse[file_index])
        impulse = df.values[0]

        pcd = pcd[start_index]
        udf = udf[start_index]

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
        for file_index, path in enumerate(self.udf):
            num_points = pd.read_pickle(self.udf[path]).shape[0]
            for start_index in range(0, num_points - self.chunk_size, self.chunk_size):
                indices.append((file_index, start_index))

        return indices

def variable_size_collate_fn(batch):
    # Batch is a list of tensors
    print(type(batch))
    print(len(batch))
    return batch

def FractureDataLoader(path):
    # df = pd.read_pickle(path)
    # print(df.head())

    # X = df.drop("distance", axis=1).values
    # y = df["distance"].values
    # # y[y > 0.3] = 0.3
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dataset = FractureDataset(path, chunk_size=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, )

    return dataloader, dataset

