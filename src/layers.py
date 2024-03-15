#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:32:57 2024

@author: filipk
"""

import torch
from torch import nn
from torch.nn import functional as F
#from attention import SelfAttention
#import math as m

import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed :int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, causal_mask = True):
        # x:(Batch_Size, Seq_len, Dim)    
        
        input_shape = x.shape
        
        batch_size, sequence_length, d_embed = input_shape
        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (Batch_size, Seq_len, Dim) -> (Batch_size, Seq_len, Dim * 3) -> 3 tensors of shape (Batch_size, Seq_len, Dim) 
        q, k, v = self.in_proj(x).chunk(3,dim=-1)
        
        # (Batch_Size, Seq_len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)
        
        # (Batch_Size, H, Seq_Len, Seq_Len)
        weight =  q @ k.transpose(-1,-2)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight ,dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
            
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim =-1)
        
        
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v
        
        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H) 
        output = output.transpose(1,2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        # (Batch_Size ,Seq_Len, Dim)
        return output


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Features, Height, Width)
        
        residue = x
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width) 
        x = x.view(n, c, h*w)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features) 
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features) 
        x = self.attention(x)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height * Width)  
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)  
        x = x.view((n,c,h,w))
        
        x += residue

        return x        

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)        
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor :
        # x: (Batch_Size, In_Channels, Height, Width)
        
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        
        return x + self.residual_layer(residue)

