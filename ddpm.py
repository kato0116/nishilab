import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from functools import partial # 関数に渡される一部の引数を指定
from einops import rearrange,repeat,reduce
from einops.layers.torch import Rearrange
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

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(dim,default(dim_out,dim),3,padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim*4,default(dim_out,dim),1)
    )

# 正規化層
class RMSNorm(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1,dim,1,1))
    def forward(self,x):
        return F.normalize(x,dim=1) * self.g * (x.shape[1]**0.5)

# 位置エンコーディング
class SinusoidalPosEmb(nn.Module):
    def __init__(self,dim,theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
    def forward(self,x):
        device = x.device
        half_dim = self.dim //2
        emb = math.log(self.theta) / (half_dim-1)
        emb = torch.exp(torch.arange(half_dim,device=device)*-emb)
        emb = x[:,None] * emb[None,:]
        emb = torch.cat((emb.sin(),emb.cos()),dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(nn.Module):
    def __init__(self,dim,dim_out,groups=8):
        super.__init__()
        self.proj = nn.Conv2d(dim,dim_out,3,padding=1)
        self.norm = nn.GroupNorm(groups,dim_out)
        self.act  = nn.SiLU()
        
    def forward(self,x,scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x*(scale+1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# model
