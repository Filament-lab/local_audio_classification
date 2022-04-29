import numpy as np
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, label_array: np.ndarray, feature_array: np.array):
        self.label_array = label_array
        self.feature_array = feature_array

    def __getitem__(self, index: int):
        return self.feature_array[index], self.label_array[index]

    def __len__(self):
        return len(self.label_array)
