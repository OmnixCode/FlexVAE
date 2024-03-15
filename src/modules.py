#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:21:46 2024

@author: filipk
"""

import torch
from torch import nn
from torch.nn import functional as F
from layers import VAE_AttentionBlock, VAE_ResidualBlock
import json
import math as m




def model_creator(structure):
    
    '''
    This function creates list of pytorch modules for passing into nn.Sequential as a list of arguments.
    Input ot this function is a list of layer codes with arguments. 
    e.g. 'C2d_3_128_3_1' is nn.Conv2D(3, 128, kernel_size=3, padding=1)
    format also accepts short notation for multiples of the same layer 
    e.g. '2*VAERB_128_128' which stands for 'VAE_ResidualBlock(128,128)->VAE_ResidualBlock(128,128)'
    '''
    
    layer_dict={'C2d' : nn.Conv2d,
                'VAERB' : VAE_ResidualBlock,
                'VAEA' : VAE_AttentionBlock,
                'GN' : nn.GroupNorm,
                'SiLU' : nn.SiLU,
                'MaxP' : nn.MaxPool2d,
                'UpS' : lambda x : nn.Upsample(None,x)
                }
    VAE_Encoder_Struct=structure
    ai_struct=[]
    for layer in range(len(VAE_Encoder_Struct)):
        if VAE_Encoder_Struct[layer].find('*') != -1:
            komp=VAE_Encoder_Struct[layer].split(sep='*')
            mult= int(komp[0])
            ltype = komp[1]
        else:
            mult = 1
            ltype = VAE_Encoder_Struct[layer]
            
        for i in range(int(mult)):
            ai_struct.append(ltype)
    
    seq_struct=[]
    for layer in ai_struct:
        components=layer.split(sep='_')
        values=[int(x) for x in components[1:]]
        seq_struct.append(layer_dict[components[0]](*values))
    return seq_struct




class VAE_Encoder(nn.Sequential):
    def __init__(self, structure_file, img_size, lat_size, reduction=[4,4], dropout_early=True, der=0.2, dropout_late=True, dlr=0.1, latent_conversion_disable = False):
        #structure_file="VAE_encoder"
        PATH_CFG = "model_structures/"+ structure_file +".mstruct"
        with open(PATH_CFG, 'r') as f:
            structure=json.load(f)
            layers_array = model_creator(structure)
            
        super(VAE_Encoder, self).__init__(*layers_array)   #create neural network
            
        
        if latent_conversion_disable == False:
            self.mu_layer = nn.Linear(int((img_size * img_size)/16), lat_size)
            self.log_var_layer = nn.Linear(int((img_size * img_size)/16), lat_size)
        self.mu= None
        self.log_var= None
        self.reduction = reduction
        self.dropout_early=dropout_early
        self.dropout_late=dropout_late
        self.latent_conversion_disable=latent_conversion_disable
    
    def reparametrize(self, mean, log_variance, noise=None):
        variance = log_variance.exp()
        stdev = variance.sqrt()
        if noise == None:
            noise = torch.randn_like(variance) 
        z = mean + stdev * noise
        return z
    
    def forward(self, x: torch.tensor, noise: torch.Tensor = None) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channel, Height / 8, Width / 8)
        
        #x = x + torch.randn_like(x) * 0.1 
        
        for module in self:
# =============================================================================
#             if ((getattr(module, 'stride', None) == (self.reduction[0],self.reduction[0]))
#             or (getattr(module, 'stride', None) == (self.reduction[1],self.reduction[1]))) :
#                 # (Padding_left, Padding_Right, Padding_Top, Padding_Bottom)
#                 x = F.pad(x,(0,1,0,1))  #WHY????
# =============================================================================
            if getattr(module, 'stride', None) == (2,2):
                # (Padding_left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x,(0,1,0,1))  #WHY????                
            if type(module) not in [torch.nn.modules.linear.Linear,torch.nn.modules.dropout.Dropout2d,torch.nn.modules.dropout.Dropout]:
                x = module(x)
            
            if (type(module) == torch.nn.modules.dropout.Dropout2d and self.dropout_early==True ) :
                x = module(x)
                
            if (type(module) == torch.nn.modules.dropout.Dropout2d and self.dropout_early==True ) :
                x = module(x)
                
        
        #assert n == 32, str(bch)+' '+str(chan)+' '+str(n)+' '+str(m)
        mean_mat, log_variance_mat = torch.chunk(x, chunks=2, dim=1)
        
        if self.latent_conversion_disable == False:
            bch, chan, n, m = mean_mat.size()
            #assert bch == 32, str(bch)+' '+str(chan)+' '+str(n)+' '+str(m)
            mean_mat = mean_mat.view(bch, chan, n * m)    
            log_variance_mat = log_variance_mat.view(bch, chan, n * m)    
            self.mu = self.mu_layer(mean_mat)
            self.log_var = self.log_var_layer(log_variance_mat)
            self.log_var = torch.clamp(self.log_var, -30, 20)
        else:
            self.log_var = torch.clamp(log_variance_mat, -30, 20)
            self.mu = mean_mat
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
# =============================================================================
#         mean, log_variance = torch.chunk(x, chunks=2, dim=1)
#         self.mu = mean
#         self.log_var= log_variance
# =============================================================================
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        #log_variance = torch.clamp(log_variance, -30, 20)
        
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)       
        #variance = log_variance.exp()
        
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)       
        #stdev = variance.sqrt()
        
        # N(0, 1) -> N(mean, variance)=X?
        # X = mean + stdev * Z
        #x = mean + stdev * noise
        x = self.reparametrize(self.mu, self.log_var, noise)
        
        # Scale the output by a constant (why? found in the original work)
        x*= 0.18215
        if self.latent_conversion_disable == False:
            bch, chan, m = x.size()
        #assert m==16
        return (x)

#old implementation
# =============================================================================
#     def forward(self, x: torch.tensor, noise: torch.Tensor) -> torch.Tensor:
#         # x: (Batch_Size, Channel, Height, Width)
#         # noise: (Batch_Size, Out_Channel, Height / 8, Width / 8)
#         
#         for module in self:
#             if getattr(module, 'stride', None) == (4,4):
#                 # (Padding_left, Padding_Right, Padding_Top, Padding_Bottom)
#                 x = F.pad(x,(0,1,0,1))  #WHY????
#             x = module(x)
#                 
#         # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
#         mean, log_variance = torch.chunk(x, chunks=2, dim=1)
#         self.mu = mean
#         self.log_var= log_variance
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
#         log_variance = torch.clamp(log_variance, -30, 20)
#         
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)       
#         variance = log_variance.exp()
#         
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)       
#         stdev = variance.sqrt()
#         
#         # N(0, 1) -> N(mean, variance)=X?
#         # X = mean + stdev * Z
#         x = mean + stdev * noise
#         
#         # Scale the output by a constant (why? found in the original work)
#         x*= 0.18215
#         return (x)
# =============================================================================
   
class VAE_Decoder(nn.Sequential):
    
    def __init__(self, structure_file, img_size, lat_size, dropout_early=True, der=0.2, dropout_late=True, dlr=0.1, latent_conversion_disable = False):
            #self.reshape_latent = nn.Linear(lat_size, (img_size * img_size)/16) 
            PATH_CFG = "model_structures/"+ structure_file +".mstruct"
            with open(PATH_CFG, 'r') as f:
                structure=json.load(f)
                layers_array = model_creator(structure)
                
            super(VAE_Decoder, self).__init__(*layers_array)   #create neural network

            
            self.conversion = nn.Linear(lat_size, int((img_size * img_size)/16))
            self.latent_conversion_disable = latent_conversion_disable
            
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        
        x/= 0.18215
        
        if self.latent_conversion_disable == False:
            x= self.conversion(x)
            b, c, nm = x.size()
            x = x.view(b, c, int(m.sqrt(nm)),int(m.sqrt(nm)))

        for module in self:
            if type(module) != torch.nn.modules.linear.Linear:
                x = module(x)
                
        # x: (Batch_Size, 3, Height, Width)
        return x
                    