import torch
import torch.nn as nn

from utils import *
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
from eval import sparse_acc
from torch.autograd import Variable
from torchviz import make_dot
import re


def subscribe(x, i):
    if isinstance(x, nn.Embedding):
        return x(torch.tensor(i)).squeeze()
    else:
        return x[i]


def naive_sim_fuser(sims, param=None, device='cuda'):
    curr_sim = None
    for i, sim in enumerate(sims):
        if sim is not None:
            sim = sim.to(device)
            if param is not None:
                sim = sim * subscribe(param, i)
            # sim = sparse_softmax(sim, 1).to(device)
            curr_sim = sim if curr_sim is None else sim + curr_sim
    return curr_sim
