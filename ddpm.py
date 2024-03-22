import math

import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator

def exists(x):
    return x is not None

def default(val,d):
    if exists(val):
        return val
    # callable(): 呼び出し可能なオブジェクトか判定
    return d() if callable(d) else d

# tupleに型を変換
def cast_tuple(t,length=1):
    if isinstance(t,tuple):
        return t
    return ((t,)*length)

# numerがdenomで割り切れるか判定
def divisible_by(numer, denom):
    return (numer%denom)==0

# 他の関数を引数にとるものなどに対して,特定の処理を行わない操作として利用可
def identity(t, *arg, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
a = [1,2,3]
my_a = cycle(a)
for _ in range(10):
    print(next(my_a))
    