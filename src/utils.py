#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
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
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        #AddGaussianNoise(0.1, 0.08),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    #num_workers / pin_memory keep the GPU fed while the CPU loads the next batch;
    #defaults keep old single-process behavior if the keys are missing from a config
    num_workers = int(getattr(args, 'num_workers', 0))
    pin_memory = bool(getattr(args, 'pin_memory', False))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader

def load_image(image_path, args):
    # Define a transformation to be applied to the image
    transform = transforms.Compose([
        SquarePad(),
        torchvision.transforms.Resize(int(args.image_size + 1/4 *args.image_size)),  # args.image_size + 1/4 *args.image_size
        #torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
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
              
              
              
def _checkpoint_payload(model, optimizer, loss, epoch, img_size, lat_size, kld_mult, ema_model=None):
    payload = {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss,
        'epoch' : epoch,
        'image_size' : img_size,
        'latent_size' : lat_size,
        'kld_mult' : kld_mult
        }
    if ema_model is not None:
        payload['ema_state_dict'] = ema_model.state_dict()
    return payload


def save_model_checkpoint(model, optimizer, loss, epoch, img_size, lat_size, kld_mult, args, ema_model=None):
    param_string = str(img_size) + '_to_' + str(lat_size) + '_kld_mult_' + str(kld_mult) + '_epoch_' + str(epoch) + "ckpt.pt"
    PATH = os.path.join("models", args.run_name, param_string)
    torch.save(_checkpoint_payload(model, optimizer, loss, epoch, img_size, lat_size, kld_mult, ema_model), PATH)

    param_string = str(img_size) + '_to_' + str(lat_size) + '_kld_mult_' + str(kld_mult) + '_epoch_' + str(epoch-1) + "ckpt.pt"
    PATH = os.path.join("models", args.run_name, param_string)
    if os.path.exists(PATH):
        os.remove(PATH)
        
        
def save_model_backup(model, optimizer, loss, epoch, img_size, lat_size, kld_mult, args, ema_model=None):
    if (((epoch+1)% args.backup_every_n_iter ==0) and (epoch !=0)):
        param_string = str(img_size) + '_to_' + str(lat_size) + '_kld_mult_' + str(kld_mult) + '_epoch_' + str(epoch) + "ckpt.pt"
        PATH = os.path.join("models", args.run_name,"backup", param_string)
        torch.save(_checkpoint_payload(model, optimizer, loss, epoch, img_size, lat_size, kld_mult, ema_model), PATH)

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



def load_model_checkpoint(model, optimizer, PATH, ema_model=None): 
    '''
    Loads model and optimizer parameters from the PATH variable.
    If ema_model is provided and the checkpoint contains ema_state_dict, it is restored too.
    '''
    #the memmory is used ineficiently... try to correct so the model is not initiated twice
    ckpt = torch.load(PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    loss = ckpt['loss']
    start_epoch = ckpt['epoch']+1
    kld_mult = ckpt['kld_mult']
    if ema_model is not None and 'ema_state_dict' in ckpt:
        ema_model.load_state_dict(ckpt['ema_state_dict'])
    model.train()
    return model, optimizer, loss, start_epoch, kld_mult


def effective_kld_weight(epoch, args):
    """
    Annealed KLD weight for the current epoch.

    - none / off: use args.kld_weight as-is
    - linear: ramp from 0 to kld_weight over kld_anneal_epochs
    - cyclical: within each cycle of length kld_anneal_cycle, ramp 0->kld_weight
      over the first half of the cycle, then stay at kld_weight (Fu et al. style)

    Starting near zero lets the model first learn to reconstruct; raising KLD later
    pulls the posterior toward N(0,1) so prior sampling works.
    """
    target = float(getattr(args, 'kld_weight', 1/4*0.01))
    mode = str(getattr(args, 'kld_anneal', 'none')).lower()
    if mode in ('none', 'off', 'false', ''):
        return target
    if mode == 'linear':
        warm = max(1, int(getattr(args, 'kld_anneal_epochs', 100)))
        return target * min(1.0, float(epoch) / float(warm))
    if mode == 'cyclical':
        cycle = max(2, int(getattr(args, 'kld_anneal_cycle', 50)))
        pos = epoch % cycle
        half = cycle // 2
        if pos < half:
            return target * (float(pos) / float(half))
        return target
    raise ValueError(f"Unknown kld_anneal mode '{mode}'. Use none, linear or cyclical.")


def build_lr_scheduler(optimizer, args, epochs):
    """
    Create the epoch-level LR scheduler selected by args.scheduler_type.
    Returns None when scheduling is disabled.
    """
    if getattr(args, 'use_scheduler', False) is False:
        return None
    stype = str(getattr(args, 'scheduler_type', 'cosine')).lower()
    if stype in ('none', 'off'):
        return None
    if stype == 'step':
        step_size = int(getattr(args, 'scheduler_step_size', 100))
        gamma = float(getattr(args, 'scheduler_gamma', 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if stype == 'cosine':
        t_max = int(getattr(args, 'cosine_t_max', 0) or epochs)
        eta_min = float(getattr(args, 'cosine_eta_min', 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max), eta_min=eta_min)
    raise ValueError(f"Unknown scheduler_type '{stype}'. Use cosine, step or none.")


#for multi GPU enviroment
class GPU_thread:
    '''
    Class for objects running on different GPUs with the copy of the original model 
    but with diferent data (sub-batch).
    '''
    
    def __init__(self,cuda_id, memory, result_queue, use_amp=False):
        self.id = cuda_id
        self.memory = memory
        self.result_queue = result_queue
        self.use_amp = use_amp
                  
    def update_state_dict(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)
        
    def predict(self, images, model):
        device_type = 'cuda' if images.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, enabled=self.use_amp):
            predicted_image = model(images)
        #mu and log_var are returned as well, so the loss can be computed over the
        #latent statistics of the whole batch and not just the sub-batch of GPU 0
        entry ={self.id : (predicted_image, model.encoder.mu, model.encoder.log_var)}
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
     
            
     

