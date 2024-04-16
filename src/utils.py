#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import os
import json       
import torchvision.transforms.functional as F
#from torch import nn

def plot_images(images, args): ##not the best plotter... change this
    plt.figure(figsize=(args.image_size, args.image_size))
    plt.imshow(torch.cat([
        torch.cat([i for i in images], dim=-1),
    ], dim=-2).permute(1, 2, 0))
    plt.show()  

def save_images(images, path):
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)
    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
# =============================================================================
#     ndarr = ndarr** 255
#     ndarr = ndarr.astype(np.uint8)
# =============================================================================
    im = Image.fromarray(ndarr)
    im.save(path)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
         
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
     
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')

    
def get_data(args):
    transforms = torchvision.transforms.Compose([
        SquarePad(),
        torchvision.transforms.Resize(int(args.image_size + 1/4 *args.image_size)),  # args.image_size + 1/4 *args.image_size
        #torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        #AddGaussianNoise(0.1, 0.08),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def load_image(image_path, args):
    # Define a transformation to be applied to the image
    transform = transforms.Compose([
        SquarePad(),
        torchvision.transforms.Resize(int(args.image_size + 1/4 *args.image_size)),  # args.image_size + 1/4 *args.image_size
        #torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_path).convert('RGB')  # Ensure that the image is in RGB format

    # Apply the transformation to the image
    tensor_image = transform(image)

    # Add an extra dimension to the tensor (batch dimension)
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def load_image2(image_path):
    # Define a transformation to be applied to the image
    transform = transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_path) # Ensure that the image is in RGB format

    # Apply the transformation to the image
    tensor_image = transform(image)

    # Add an extra dimension to the tensor (batch dimension)
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def check_nan_inf(value, name="Value"):
    """
    This function is self-explanatory.
    It checks if tensor or float is NaN or inf valued 
    """
    if isinstance(value, torch.Tensor):
        # Check for NaN values
        if torch.isnan(value).any():
            print(f"{name} (tensor) contains NaN values.")
            return 'Error'

        # Check for Inf values
        if torch.isinf(value).any():
            print(f"{name} (tensor) contains Inf values.")
            return 'Error'
        
    elif isinstance(value, (float, int)):
        # Check for NaN values
        if torch.isnan(torch.tensor(value)).item():
            print(f"{name} (float) is NaN.")
            return 'Error'
        # Check for Inf values
        if torch.isinf(torch.tensor(value)).item():
            print(f"{name} (float) is Inf.")
            return 'Error'
    else:
        print(f"Unsupported type for {name}: {type(value)}")

def check_parameters_for_naninf(parameters : dict):
    for parameter in parameters:
          if check_nan_inf(parameters[parameter], parameter)=='Error':
              print(check_nan_inf(parameters[parameter], parameter))
              
              
              
def save_model_checkpoint(model, optimizer, loss, epoch, img_size, lat_size, kld_mult, args):
    param_string = str(img_size) + '_to_' + str(lat_size) + '_kld_mult_' + str(kld_mult) + '_epoch_' + str(epoch) + "ckpt.pt"
    PATH = os.path.join("models", args.run_name, param_string)
    torch.save({
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss,
        'epoch' : epoch,
        'image_size' : img_size,
        'latent_size' : lat_size,
        'kld_mult' : kld_mult
        }, PATH)

    param_string = str(img_size) + '_to_' + str(lat_size) + '_kld_mult_' + str(kld_mult) + '_epoch_' + str(epoch-1) + "ckpt.pt"
    PATH = os.path.join("models", args.run_name, param_string)
    if os.path.exists(PATH):
        os.remove(PATH)
        
        
def save_model_backup(model, optimizer, loss, epoch, img_size, lat_size, kld_mult, args):
    if (((epoch+1)% args.backup_every_n_iter ==0) and (epoch !=0)):
        param_string = str(img_size) + '_to_' + str(lat_size) + '_kld_mult_' + str(kld_mult) + '_epoch_' + str(epoch) + "ckpt.pt"
        PATH = os.path.join("models", args.run_name,"backup", param_string)
        torch.save({
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : loss,
            'epoch' : epoch,
            'image_size' : img_size,
            'latent_size' : lat_size,
            'kld_mult' : kld_mult
            }, PATH)

        param_string = str(img_size) + '_to_' + str(lat_size) + '_kld_mult_' + str(kld_mult) + '_epoch_' + str(epoch) + "ckpt.txt"
        PATH = os.path.join("models", args.run_name,"backup", param_string)
        with open(PATH, 'w+') as f:
            json.dump(args.__dict__, f, indent=2)
            
            
def setup_logging(run_name, names=["models","results", "samples"]):
    """
    Makes folders for model, sampling and log saves.
    """
    for folder_name in names:
        os.makedirs(folder_name, exist_ok=True)
        os.makedirs(os.path.join(folder_name, run_name), exist_ok=True)

    os.makedirs(os.path.join("models", run_name, "backup"), exist_ok=True)



def load_model_checkpoint(model, optimizer, PATH): 
    '''
    Loads model and optimizer parameters from the PATH variable
    '''
    #the memmory is used ineficiently... try to correct so the model is not initiated twice
    ckpt = torch.load(PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    loss = ckpt['loss']
    start_epoch = ckpt['epoch']+1
    kld_mult = ckpt['kld_mult']
    model.train()
    return model, optimizer, loss, start_epoch, kld_mult


#for multi GPU enviroment
class GPU_thread:
    '''
    Class for objects running on different GPUs with the copy of the original model 
    but with diferent data (sub-batch).
    '''
    
    def __init__(self,cuda_id, memory, result_queue):
        self.id = cuda_id
        self.memory = memory
        self.result_queue = result_queue
                  
    def update_state_dict(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)
        
    def predict(self, images, model):
        predicted_image = model(images)
        entry ={self.id : predicted_image}
        self.result_queue.put(entry)
 
class Configs:
    def __init__(self, init_dict={}):
            self._variables = init_dict

    def __getattr__(self, name):
        if name in self._variables:
            return self._variables[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name != '_variables':
            self._variables[name] = value
        else:
            super().__setattr__(name, value)           
     
            
     

