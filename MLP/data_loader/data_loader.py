import random
import os
import pandas as pd
from deltaconv.experiments.datasets.shape_seg import read_obj
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class FractureDataset(Dataset):
    def __init__(self, root_directory, dataset_type="train"):
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

        # pcd = pcd[start_index: start_index + self.chunk_size]
        # udf = udf[start_index: start_index + self.chunk_size]
        # label_gt = label_gt[start_index: start_index + self.chunk_size]
        # label_edge = label_edge[start_index: start_index + self.chunk_size]

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

    def get_GT_size(self):
        return len(self.udf)

    def get_random_GT(self):
        index = random.randint(0, self.files_used - 1)
        return self.get_GT(index)

    def _prepare_data_indices(self):
        indices = []
        for file_index, path in self.udf.items():

            num_points = pd.read_pickle(path).shape[0]
            # for start_index in range(0, num_points - self.chunk_size, self.chunk_size):
            #     indices.append((file_index, start_index))
            indices.append((file_index, 0))
        return indices


def variable_size_collate_fn(batch):
    # Batch is a list of tensors
    print(type(batch))
    print(len(batch))
    return batch


def FractureDataLoader(path, type, batch_size=1):
    # df = pd.read_pickle(path)
    # print(df.head())

    # X = df.drop("distance", axis=1).values
    # y = df["distance"].values
    # # y[y > 0.3] = 0.3
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dataset = FractureDataset(path, dataset_type=type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, )

    return dataloader, dataset





import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data


class FractureGeomDataset(InMemoryDataset):

    # set url
    # set folders for MIT dataset

    # root should be MLP/datasets
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None, dataset_name="bunny"):
        self.used_dataset_name = dataset_name
        self.udf = {}
        self.impulse = {}
        self.files_used = 0
        self.mesh_vertices = []
        self.mesh_triangles = []
        self.split = split
        for file in os.listdir(os.path.join(root, self.used_dataset_name)):
            if not file.split(".")[-1] == "pkl":
                continue
            without_filetype = file.split('.')[0]
            if len(without_filetype.split('_')) == 1:
                index = int(without_filetype)
                self.udf[index] = os.path.join(root, self.used_dataset_name, file)
                self.files_used += 1
            else:
                index = int(without_filetype.split('_')[0])
                self.impulse[index] = os.path.join(root, self.used_dataset_name, file)

        super(FractureGeomDataset, self).__init__(root, transform, pre_transform)
        # set processed path
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'validation':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        elif split == 'full':
            path = self.processed_paths[3]
        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

        # returns name to zip file containing dataset

    @property
    def processed_file_names(self):
        return [f'{self.used_dataset_name}_training.pt', f'{self.used_dataset_name}_test.pt', f'{self.used_dataset_name}_validate.pt', f'{self.used_dataset_name}_full.pt']
        # return list of file names containing training and test data


    def process(self):
        print("started processing dataset")

        # create list containing all data named data_list
        data_list = []
        data_list_test = []
        data_list_validate = []
        data_list_full = []

        # read faces from obj file
        mesh = read_obj(os.path.join(self.root, self.used_dataset_name, self.used_dataset_name + ".obj"))
        self.mesh_vertices = mesh.pos.cpu().numpy()
        self.mesh_triangles = mesh.face.cpu().numpy().T
        # loop over all files in dir
        for file_index in tqdm(range(self.files_used)):
            data = read_obj(os.path.join(self.root, self.used_dataset_name, self.used_dataset_name + ".obj"))
            # visualize_mesh_from_data_obj(data)

            filename = self.udf[file_index]
            df = pd.read_pickle(filename)
            # df = df.head(3)
            # data.pos = data.pos[:3]
            # data.face = torch.tensor([[0],[1],[2]])
            udf = df["distance"].values
            label_gt = df["label"].values
            label_edge = df['edge_labels'].values

            df = pd.read_pickle(self.impulse[file_index])
            impulse = df.values[0]

            data.y = torch.tensor(udf, dtype=torch.float32)
            data.gt_label = torch.tensor(label_gt, dtype=torch.int64)
            data.impulse = torch.tensor(impulse, dtype=torch.float32)
            data.edge_label = torch.tensor(label_edge, dtype=torch.int64)
            # data.x = torch.tensor(impulse, dtype=torch.float32).repeat(udf.shape[0], 1)
            data.x = torch.cat((data.pos, torch.tensor(impulse, dtype=torch.float32).repeat(udf.shape[0], 1)), dim=1)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            if file_index < self.files_used * 0.7:
                data_list.append(data)
            elif file_index < self.files_used * 0.85:
                data_list_test.append(data)
            else:
                data_list_validate.append(data)
            data_list_full.append(data)
            # visualize_mesh_from_data_obj(data)
        if self.split == 'train':
            torch.save(self.collate(data_list), self.processed_paths[0])
        if self.split == 'validation':
            torch.save(self.collate(data_list_validate), self.processed_paths[1])
        if self.split == 'test':
            torch.save(self.collate(data_list_test), self.processed_paths[2])
        if self.split == 'full':
            torch.save(self.collate(data_list_full), self.processed_paths[3])

        # save data_list to disk, after applying self.collate to the data_list

        # rmtree all unnecessary files






def visualize_mesh_from_data_obj(data):
    import polyscope as ps
    ps.set_window_size(1920, 1080)
    ps.init()

    pos = data.pos
    face = data.face

    gt = data.y.cpu().numpy()

    vertices = pos.cpu().numpy()
    faces = face.cpu().numpy().T

    ps.register_surface_mesh("UDF mesh", vertices, faces, smooth_shade=True)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("GT distance scalar", gt, defined_on="vertices",
                                                        enabled=True)

    ps.look_at((0., 0., 2.5), (0, 0, 0))
    ps.show()
