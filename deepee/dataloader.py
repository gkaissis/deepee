from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import numpy as np
from typing import Generator


class UniformDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        """Convenience DataLoader with uniform subsampling without replacement.
        Suitable for use with the Gaussian DP privacy accounting used in the library.

        Args:
            dataset (Dataset): A Dataset instance
            batch_size (int): The desired batch size.
            num_workers (int, optional): How many workers to use. Defaults to 0.
            May result in deadlocks or similar undefined behaviour, in which case it is
            recommended to set this to 0. Compare https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading. When using CUDA, it should be set
            to 0 and pin_memory should be set to True.
            pin_memory (bool, optional): [description]. Defaults to True. Use pinned memory
            for allocating the dataset. Recommended when CUDA is used.
        """

        super().__init__(
            dataset,
            batch_sampler=UniformWORSubsampler(dataset=dataset, batch_size=batch_size),
            pin_memory=pin_memory,
            num_workers=num_workers,
        )


class UniformWORSubsampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int):
        """Uniform random subsampler without replacement.
        Compare https://arxiv.org/pdf/1807.01647.pdf and
        https://arxiv.org/pdf/1808.00087.pdf.

            Args:
                dataset (Dataset): A torch Dataset instance.
                batch_size (int): The desired batch size.
        """
        self.sample_size = len(dataset)  # type: ignore
        self.batch_size = batch_size

    def __iter__(self) -> Generator:
        for _ in range(len(self)):
            yield np.random.choice(self.sample_size, self.batch_size, replace=False)

    def __len__(self) -> int:
        return self.sample_size // self.batch_size
