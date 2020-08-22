import torch
from random import Random
import torch.distributed as dist

class DistributedTraining:
    def __init__(self):
        pass
    
    def average_gradients(self, model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        
    def partition_dataset(self, X, batch_size):

        dataset = X
        size = dist.get_world_size()
        bsz = batch_size / float(size)
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = self.DataPartitioner(dataset, partition_sizes)
        rank_partitions = partition.use()
        partition = self.Partition(dataset, rank_partitions[dist.get_rank()])
        train_loader = torch.utils.data.DataLoader(partition, batch_size=int(bsz), shuffle=True)
        return train_loader, bsz
    
    class Partition(object):

        def __init__(self, data, index):
            self.data = data
            self.index = index

        def __len__(self):
            return len(self.index)

        def __getitem__(self, index):
            data_idx = self.index[index]
            return self.data[data_idx]


    class DataPartitioner(object):

        def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
            self.data = data
            self.partitions = []
            self.sizes = sizes
            self.seed = seed  

        def use(self):
            rng = Random()
            rng.seed(self.seed)
            data_len = len(self.data)
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)

            for frac in self.sizes:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]
            return self.partitions
