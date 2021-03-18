from deepee import UniformDataLoader
from torch.utils.data import Dataset
import torch


class SimpleDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.arange(1, 100, 1, dtype=torch.int)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


ds = SimpleDataset()

dl = UniformDataLoader(ds, 50)


def test_dataloader():
    for item in dl:
        assert (
            len(set(item)) == 50
        )  # always returns correct batch size and never the same item twice
