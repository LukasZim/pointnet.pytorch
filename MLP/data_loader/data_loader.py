import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class FractureData(Dataset):
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            if scale_data:
                X = StandardScaler().fit_transform(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def FractureDataLoader(path):
    df = pd.read_csv(path)
    print(df.head())

    X = df.drop("distance", axis=1).values
    y = df["distance"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dataset = FractureData(X, y, scale_data=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    return dataloader, X, y, X_train, y_train, X_test, y_test
