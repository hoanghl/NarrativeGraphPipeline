from random import sample
import math

from torch.utils.data.sampler import Sampler


class CustomSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, size_dataset, n_shards):
        self.size_dataset = size_dataset
        self.n_shards = n_shards

        self.indices = None
        self.shuffle()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return self.size_dataset

    def shuffle(self):
        size_shard = math.ceil(self.size_dataset / self.n_shards)

        shard_order = sample(range(self.n_shards), self.n_shards)

        self.indices = []
        for nth_shard in shard_order:
            start = size_shard * nth_shard
            end = (
                start + size_shard
                if nth_shard != max(shard_order)
                else self.size_dataset
            )
            indices = sample(range(start, end), end - start)

            self.indices.extend(indices)
