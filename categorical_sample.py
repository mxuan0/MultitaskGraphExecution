from torch.distributions.categorical import Categorical
import torch

def _sample(p:torch.Tensor):
    dist = Categorical(p)
    return dist.sample()