from torch.distributions.categorical import Categorical
import torch

def categorical_sample(p:torch.Tensor):
    dist = Categorical(p)
    return dist.sample()

def batch_sample(batches:list, sizes:torch.Tensor, task_list:list):
    task_to_batch = {}
    for i in range(len(batches)):
        if sizes[i] != 0:
            task_to_batch[task_list[i]] = batches[i][:sizes[i]]
    return task_to_batch