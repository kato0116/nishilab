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

# numが平方根を持つか判定
def has_int_squareroot(num):
    return (math.sqrt(num)**2) == num

# numをdivisorで指定されたサイズのグループに分割 ex)num:9,div:3 -> 3,3,3
def num_to_groups(num,divisor):
    groups = num//divisor     # グループ数
    remainder = num % divisor # あまり
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# imageを指定のtypeに変換
def convert_image_to_fn(img_type,image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# 0~1の値を-1~1に変換 (正規化)
def normalize_to_neg_one_to_one(img):
    return img*2-1
# -1~1の値を0~1に変換
def unnormalize_to_zero_to_one(t):
    return (t+1)*0.5

