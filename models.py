import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numbers
from einops import rearrange


from attention import CrossAttention
from upsampling_downsampling import Downsample, Upsample




# class Attention(nn.Module):
#     def __init__(self, dim):
#         super(Attention, self).__init__()

#         self.conv_i = nn.Conv2d(dim, dim, 1)        
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
#         self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

#         self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#         self.conv_f = nn.Conv2d(dim, dim, 1)

#         self.activation = nn.GELU()

#     def forward(self, x):

#         # 1x1 conv
#         x = self.conv_i(x)
#         # GELU
#         x = self.activation(x)

#         u = x.clone()
#         attn = self.conv0(x)

#         attn_0 = self.conv0_1(attn)
#         attn_0 = self.conv0_2(attn_0)

#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)

#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2)
#         attn = attn + attn_0 + attn_1 + attn_2

#         attn = self.conv3(attn)

#         out = attn * u

#         out = self.conv_f(out)
#         return out
        
# class FFN(nn.Module):
#     def __init__(self, dim):
#         super(FFN, self).__init__()
#         self.conv1 = nn.Conv2d(dim, dim, 1)
#         self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#         self.activation = nn.GELU()
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x

# class MSCAN(nn.Module):
#     def __init__(self, dim=64):
#         super(MSCAN, self).__init__()
#         self.attention = Attention(dim)
#         self.ffn = FFN(dim)
#         self.norm = nn.BatchNorm2d(num_features = dim)

#     def forward(self, x):
#         u = x.clone()
#         u = self.norm(u)
#         u = self.attention(u)
#         y = u+x
#         u = self.norm(u)
#         u = self.ffn(u)
#         return u+y

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class MSCAN(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(MSCAN, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class CC_Module(nn.Module):

    def __init__(self):
        super(CC_Module, self).__init__()   

        print("Color correction module for underwater images")

        # Convolution layers that maps  n*3*256*256 to  n*64*256*256
        self.convi = nn.Sequential(nn.Conv2d(3, 64, 3, padding = 1), nn.BatchNorm2d(num_features=64), nn.PReLU())
        self.convf = nn.Sequential(nn.Conv2d(3, 64, 3, padding = 1), nn.BatchNorm2d(num_features=64), nn.PReLU())
        self.convd = nn.Sequential(nn.Conv2d(3, 64, 3, padding = 1), nn.BatchNorm2d(num_features=64), nn.PReLU())
                
        # Define your encoder blocks here
        # N*64*256*256
        self.ei1 = MSCAN(64,1)
        self.ed1 = MSCAN(64,1)
        self.ef1 = MSCAN(64,1)
        self.ca11 = CrossAttention(64,64)
        self.ca12 = CrossAttention(64,64)
        self.downi1 = Downsample(64)
        self.downd1 = Downsample(64)
        # N*128*128*128
        self.ei2 = MSCAN(128,2)
        self.ed2 = MSCAN(128,2)
        self.ef2 = MSCAN(128,2)
        self.ca21 = CrossAttention(128,128)
        self.ca22 = CrossAttention(128,128)
        self.ca23 = CrossAttention(128,128)
        self.downi2 = Downsample(128)
        self.downd2 = Downsample(128)
        # N*256*64*64
        self.ei3 = MSCAN(256,4)
        self.ed3 = MSCAN(256,4)
        self.ef3 = MSCAN(256,4)
        self.ca31 = CrossAttention(256,256)
        self.ca32 = CrossAttention(256,256)
        self.ca33 = CrossAttention(256,256)
        self.downi3 = Downsample(256)
        self.downd3 = Downsample(256)
        # N*512*32*32
        self.ei4 = MSCAN(512,8)
        self.ed4 = MSCAN(512,8)
        self.ef4 = MSCAN(512,8)
        self.ca41 = CrossAttention(512,512)
        self.ca42 = CrossAttention(512,512)
        self.ca43 = CrossAttention(512,512)
        
        self.d1 = MSCAN(512,8)
        self.up1 = Upsample(512)
        # N*256*64*64
        self.d2 = MSCAN(256,4)
        self.up2 = Upsample(256)
        # N*128*128*128
        self.d3 = MSCAN(128,2)
        self.up3 = Upsample(128)
        # N*64*256*256
        self.conv_f = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, img, depth):
        # Pass images, fft, and depth maps through their respective encoder blocks
        # N*3*256*256
        imgc = self.convi(img)
        depthc = self.convd(depth)
        # fftc = self.convf(fft)
        
        # N*64*256*256
        img1 = self.ei1(imgc)
        # fft1 = self.ef1(fftc)
        depth1 = self.ed1(depthc)
        ca_id1 = self.ca11(img1, depth1)
        # ca1 = self.ca12(ca_temp, fft1)
        
        img1d = self.downi1(img1)
        # fft1d = self.downsampling(fft1)
        depth1d = self.downd1(depth1)
        # ca1d = self.downsampling(ca1)
        
        # N*64*128*128
        img2 = self.ei2(img1d)
        # fft2 = self.ef2(fft1d)
        depth2 = self.ed2(depth1d)
        ca_id2 = self.ca21(img2, depth2)
        # ca_temp = self.ca22(ca_temp, depth2)
        # ca2 = self.ca23(ca_temp, ca1d)
        
        img2d = self.downi2(img2)
        # fft2d = self.downsampling(fft2)
        depth2d = self.downd2(depth2)
        # ca2d = self.downsampling(ca2)
        
        # N*64*64*64
        img3 = self.ei3(img2d)
        # fft3 = self.ef3(fft2d)
        depth3 = self.ed3(depth2d)
        ca_id3 = self.ca31(img3, depth3)
        # ca_temp = self.ca32(ca_temp, depth3)
        # ca3 = self.ca33(ca_temp, ca2d)
        
        img3d = self.downi3(img3)
        # fft3d = self.downsampling(fft3)
        depth3d = self.downd3(depth3)
        # ca3d = self.downsampling(ca3)
        
        # 32*32
        img4 = self.ei4(img3d)
        # fft4 = self.ef4(fft3d)
        depth4 = self.ed4(depth3d)
        ca_id4 = self.ca41(img4, depth4)
        # ca_temp = self.ca42(ca_temp, depth4)
        # ca4 = self.ca43(ca_temp, ca3d)
        
        #32*32
        # out = self.d1(ca4)
        out = self.d1(ca_id4)
        # out += img4
        
        out = self.up1(out)
        out += ca_id3
        
        #64*64
        out = self.d2(out)
        # out += img3
        
        out = self.up2(out)
        out += ca_id2
        
        # 128*128
        out = self.d3(out)
        # out += img2
        
        out = self.up3(out)
        out += ca_id1
        # 256*256
        # out += img1

        out = self.conv_f(out)
        out += img
        
        return out
