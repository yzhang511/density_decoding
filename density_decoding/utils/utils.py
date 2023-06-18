"""General utility functions."""

import random
import torch
import numpy as np


def safe_log(x, minval=1e-8):
    return torch.log(x + minval)


def safe_divide(x, y):
    return torch.clip(x / y, min = 0, max = 1)

def get_odd_number(number):
    if number % 2 == 1:
        return number
    else:
        return number + 1


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_default_dtype(torch.double)
    
    
def to_device(x, device):
    return torch.tensor(x).to(device)





    
    