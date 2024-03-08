import math

import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator

def exists(x):
    return x is not None

print(exists(3))