from random import sample
import multiprocessing


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
        size_shard = int(self.size_dataset / self.n_shards)

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


class ParallelHelper:
    def __init__(
        self,
        f_task: object,
        data: list,
        data_allocation: object,
        n_workers,
        desc=None,
        show_bar=False,
    ):
        self.n_data = len(data)
        self.show_bar = show_bar

        self.queue = multiprocessing.Queue()
        if self.show_bar:
            self.pbar = tqdm(total=self.n_data, desc=desc)

        self.jobs = list()
        for ith in range(n_workers):
            lo_bound = ith * self.n_data // n_workers
            hi_bound = (
                (ith + 1) * self.n_data // n_workers
                if ith < (n_workers - 1)
                else self.n_data
            )

            p = multiprocessing.Process(
                target=f_task,
                args=(data_allocation(data, lo_bound, hi_bound), self.queue),
            )
            self.jobs.append(p)

    def launch(self) -> list:
        """
        Launch parallel process
        Returns: a list after running parallel task

        """
        dataset = []

        for job in self.jobs:
            job.start()

        cnt = 0
        while cnt < self.n_data:
            while not self.queue.empty():
                dataset.append(self.queue.get())
                cnt += 1

                if self.show_bar:
                    self.pbar.update()

        if self.show_bar:
            self.pbar.close()

        for job in self.jobs:
            job.terminate()

        for job in self.jobs:
            job.join()

        return dataset
