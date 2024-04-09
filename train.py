#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:40:59 2023

@author: filipk
"""
import os, sys

os.chdir("/home/filipk/Desktop/Python Projects/FlexVAE/")
sys.path.append('src/') #adds the src folder to lib path


from modules import  VAE_Encoder, VAE_Decoder
import torch

from torch import nn
from torch import optim
from torch.nn import functional as F 
#from torch.profiler import profile, record_function, ProfilerActivity

import argparse

from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

import numpy as np

from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
import math
from utils import get_data, save_images, load_image, check_nan_inf, check_parameters_for_naninf, save_model_checkpoint, save_model_backup
from utils import setup_logging
from utils import GPU_thread
from utils import load_model_checkpoint
from utils import Configs
from inference_eval import encode_from_folder

import gc #garbage collector

from collections import OrderedDict
import pprint #for nice printing of JSON config file



class VAE(nn.Module):
    def __init__(self, VAE_encoder, VAE_decoder, args):
        super().__init__()
        self.encoder = VAE_encoder(args.encoder_struct, args.image_size, args.lat_size, latent_conversion_disable= args.latent_conversion_disable)
        self.decoder = VAE_decoder(args.decoder_struct, args.image_size, args.lat_size, latent_conversion_disable= args.latent_conversion_disable)
        self.latent_conversion_disable = args.latent_conversion_disable
    def forward(self, x, noise=None):
        x=self.encoder(x, noise)
        x=self.decoder(x)
        return x
    
    def loss_function(self,input_image, reconstructed_image, epoch=0, kld_mult=0, ssim_metrics = True, alpha=0.5, beta=0.5,sparse_metrics = False, lambd = 1):
        #kld_weight=1
        kld_weight=(epoch / 100) - int(epoch / 100) #bilo0.03
        kld_weight=1/4*0.01 #bilo 10000 poslednje 1/10
        #bilo10
        recons_loss = 400*F.mse_loss(reconstructed_image, input_image)#*128*128*3 #change to 10, 20 gave interesting stuff#last=5 @bilo 100
        if ssim_metrics == True:
            ssim_loss = SSIM(reconstructed_image, input_image)
        else:
            alpha=1
        
        sparse_loss = lambd *torch.sum(torch.abs(self.encoder.mu))
        
        """
        Different reductions for two different latent space dimensions. If latent_conversion_disable == True that means that we
        keep the mean and log_variance of the lattent as a 2d tensors. If it is True, we will use nn.linear layer to convert it to 1d-tensor.
        """
        if self.latent_conversion_disable ==False:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + self.encoder.log_var - self.encoder.mu ** 2 -  self.encoder.log_var.exp(), dim = (1,2)), dim = 0)
        else:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + self.encoder.log_var - self.encoder.mu ** 2 -  self.encoder.log_var.exp(), dim = (1,2,3)), dim = 0)
        if self.latent_conversion_disable ==False:
            b,c,ld = self.encoder.log_var.size()
        else:
            b,c,l,d = self.encoder.log_var.size()
            ld=l*d
        kld_loss= kld_loss/(c*ld)
        sparse_loss= sparse_loss/(b*c*ld)
        
        #activate sparse 1809 epoch
        
        #loss = alpha*recons_loss + 0*beta*(1-ssim_loss) + kld_weight * kld_loss + 20*sparse_loss #0.3 gore
        loss = alpha*recons_loss + 0*beta*(1-ssim_loss) + kld_weight * kld_loss + 0*sparse_loss #0.3 gore
        loss = alpha*recons_loss + ssim_metrics*beta*(1-ssim_loss) + kld_weight * kld_loss + sparse_metrics*0*sparse_loss
        #loss = alpha*recons_loss +20*beta*(1-ssim_loss) #0.3 gore
        #loss=alpha*recons_loss + kld_weight * kld_loss.to(torch.device('cuda:0'))
        #return{'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        return{'loss': loss, 'Reconstruction_Loss':(alpha*recons_loss.detach()),'SSIM_Loss':(beta*(1-ssim_loss.detach())),'Sparse_Loss':20*sparse_loss, 'KLD':(kld_weight * kld_loss).detach()}
    
    def loss_function_correct9():
        pass
    
    def sample(self, n_samples, lat_size):
        noise = torch.mul(torch.randn((n_samples, 4, lat_size)).to(torch.device('cuda:0')), 1)
        x = self.decoder(noise)
        return(x)
    
    def sample2(self, n_samples, lat_size):
        noise = torch.mul(torch.randn((n_samples, 4, lat_size, lat_size)).to(torch.device('cuda:0')), 1)
        x = self.decoder(noise)
        return(x)
    
    def encode(self, image):
        z=self.encoder(image)
        return(z)

    
    def __del__(self):  
        gc.collect()
        torch.cuda.empty_cache()







      

#time 3:56
import threading
from queue import Queue
result_queues = Queue()


import signal
# Define pbar globally
pbar = None
resize_triggered = False


def clear_terminal():
    # Clear the entire terminal window
    os.system('cls' if os.name == 'nt' else 'clear')

def handle_resize(signal, frame):
    global pbar, resize_triggered
    # Check if resize has already been triggered
    if not resize_triggered:
        # Set flag to indicate resize has been triggered
        resize_triggered = False
        # Clear the entire terminal window
        clear_terminal()
        # Redraw tqdm progress bar
        if pbar:
            pprint.pprint(config.__dict__) #print config file with settings
            print('\n')
            pbar.refresh()



signal.signal(signal.SIGWINCH, handle_resize)
#epoch1104 promena padda
#import copy

def train(args): 
    global pbar, resize_triggered
    pprint.pprint(args.__dict__) #print config file with settings
    print('\n')
    setup_logging(args.run_name)
    #device=args.device
    dataloader = get_data(args)
    #model = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder, args).to(torch.device('cuda:0'))

    
    #find number of GPUs and their rounded memmory in GBs (also the ratio of memories in regards to the smallest GPU)
    num_gpu = torch.cuda.device_count() #Number of available compute devices
    
    if args.multi_GPU==False: #Use only one GPU
        num_gpu=1
        
    mem_gpu=[np.rint(torch.cuda.get_device_properties(x).total_memory/(np.power(1024,3))) for x in range(num_gpu)]
    mem_ratio=[x/np.min(mem_gpu) for x in mem_gpu]
    mem_fraction=[x/np.sum(mem_ratio) for x in mem_ratio]

    #create model on each GPU
    models=[]
    for i in range(num_gpu):
        models.append(VAE(VAE_Encoder, VAE_Decoder, args).to(torch.device('cuda:'+str(i)))) 
 
    optimizer = optim.AdamW(models[0].parameters(), lr = args.lr)
    #optimizer = optim.NAdam(model.parameters(), lr = 3e-4)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    start_epoch = 0
    kld_mult=0.01
 
    
    if args.resume == True:
        models[0], optimizer, loss, start_epoch, kld_mult = load_model_checkpoint(models[0], optimizer, args.resume_path)
        if args.reinit_optim == True:
            optimizer = optim.AdamW(models[0].parameters(), lr = 3e-4)
        #optimizer = optim.NAdam(model.parameters(), lr = 2e-4)

    if args.useEMA == True:
        ema_model = torch.optim.swa_utils.AveragedModel(models[0], multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    
    if args.ReduceLROnPlateau == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
    if args.use_scheduler == True:   
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        
        logging.info(f"Starting epoch {epoch}:")
        
        #custom_bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        pbar = tqdm(dataloader,file=sys.stdout, leave=True, dynamic_ncols=True, smoothing=0, disable=False, position=start_epoch-epoch)    
        total=0
        rec_loss=0
        kld_loss=0
        sparse_loss=0
        simm_loss=0
        loss_dict= {
                    "Reconstruction_Loss" : rec_loss,
                    "KLD" : kld_loss,
                    "Sparse_Loss" : sparse_loss,
                    "SSIM_Loss" : simm_loss
                    }
        




        
        for batch_idx, (images, _) in enumerate(pbar):
            
            #load wights and parameters from the main model (models[0])
            for i in range(1, num_gpu):
                models[i].load_state_dict(models[0].state_dict())
                
            #model2.load_state_dict(models[0].state_dict())
            
            
            images = images.to(torch.device('cuda:0'))
            #noise = torch.randn((images.size(0), 4, args.lat_size)).to(torch.device('cuda:0'))
            #noise = torch.randn((images.size(0), 4, args.lat_size))
            batch_size = np.array(images.size()[0])
            chunk_sizes= np.round(batch_size * mem_fraction)
            #print(chunk_sizes)
            #print('eeeee')
            #images1, images2= torch.split(images, dim=0, split_size_or_sections=[images.size()[0]-images.size()[0]//3, images.size()[0]//3])
            
            #split image batch into sub-batch for each GPU
            chunks = torch.split(images, dim=0, split_size_or_sections=list(chunk_sizes.astype(int)))
            
            

          #  assert images1.size()[0]==8, "List1 must not be empty " +str(images1.size()[0])
         #   assert images2.size()[0]==4, "List2 must not be empty "+str(images2.size()[0])
            

            
            
            #spin up thread instances and collect data from trheads
            thread_instances=[]
            thread=[]
            for i in range(num_gpu):
                thread_instances.append(GPU_thread(cuda_id=i, memory=mem_gpu[i], result_queue=result_queues))
                thread.append(threading.Thread(target=thread_instances[i].predict, args=(chunks[i].to(torch.device('cuda:'+str(i))), models[i])))
                thread[i].start()
                
            for i in range(num_gpu):
                thread[i].join()
                
            """
            Get the results from the thread queues and append them into one big batch again
            """
            sequence=[None]*num_gpu
            for i in range(num_gpu):
                sub_batch_dict=result_queues.get()
                number=set(sub_batch_dict).pop()
                sequence[number]=sub_batch_dict[number].to(torch.device('cuda:0'))
                
            predicted_image = torch.cat(sequence ,dim=0)



            """
            Getting the loss term and also different parts of the loss for printing out.
            """
            loss_params = models[0].loss_function(images, predicted_image, epoch, kld_mult)
            loss = loss_params['loss']
            
            for loss_type in loss_dict:
                loss_dict[loss_type] += loss_params[loss_type].item()

            total += loss.item() 

         
            
# =============================================================================
#             total += loss.item() 
#             rec_loss += loss_params['Reconstruction_Loss'].item() 
#             kld_loss += loss_params['KLD'].item() 
#             sparse_loss += loss_params['Sparse_Loss'].item() 
#             simm_loss += loss_params['SSIM_Loss'].item() 
# =============================================================================
            
            """
            Check if some of the losses become NaN or inf valued
            """
            check_parameters_for_naninf(loss_dict)



            

            
# =============================================================================
#             print(batch_idx)
# =============================================================================
            
            if ((batch_idx // args.batch_accum)+1)* args.batch_accum <= l: #izmenio na <= sa <
                if check_nan_inf(loss/args.batch_accum)=='Error':
                    print(loss_params['loss'])
                    print(batch_idx)
                    print('lossdiv1 je problem')  
                loss=loss/args.batch_accum
            else:
                if check_nan_inf(loss/(l%args.batch_accum))=='Error':
                    print(loss_params['loss'])
                    print(batch_idx)
                    print('lossdiv2 je problem')                  
                loss=loss/(l%args.batch_accum)
            
            
            
            loss.backward()
            
            div= batch_idx+1
            if ((batch_idx + 1) % args.batch_accum == 0) or (batch_idx + 1 == l): 
                optimizer.step()
                optimizer.zero_grad()
                logger.add_scalar("Total_loss", total/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
                
                for loss_type in loss_dict:
                  logger.add_scalar(loss_type,  loss_dict[loss_type]/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )  
# =============================================================================
#                 logger.add_scalar("Reconstruction_loss",  rec_loss/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
#                 logger.add_scalar("KLD_loss", kld_loss/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
#                 logger.add_scalar("Sparse_loss", sparse_loss/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
#                 logger.add_scalar("SSIM_loss", simm_loss/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
# =============================================================================
            
           #pbar.set_postfix(MSE=loss.item())
            od = OrderedDict()
            od['Epoch'] = epoch
            for loss_type in loss_dict:
                od[loss_type] = loss_dict[loss_type]/div
            od['learn_rate'] = scheduler2.get_last_lr()
            pbar.set_postfix(od)
            #pbar.set_postfix({'Epoch' : epoch, 'Total_loss':total/div,'rec_loss': loss_dict["Reconstruction_Loss"]/div,'KLD_loss':loss_dict["KLD"]/div,'Sparse':loss_dict["Sparse_Loss"]/div,'SIMM_loss':loss_dict["SSIM_Loss"]/div,'learn_rate':scheduler2.get_last_lr()})
            
        if args.ReduceLROnPlateau == True:
            scheduler.step(loss)
            
        if args.useEMA == True:
            ema_model.update_parameters(model)
            
        scheduler2.step()    
            
        save_images(images.detach(), os.path.join("results", args.run_name, f"{epoch}_orig.jpg"))
        save_images(predicted_image.detach(), os.path.join("results", args.run_name, f"{epoch}.jpg"))
        #torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

        save_model_checkpoint(models[0], optimizer, loss, epoch, images.size()[-1], args.lat_size, kld_mult, args)
        save_model_backup(models[0], optimizer, loss, epoch, images.size()[-1], args.lat_size, kld_mult, args)
        
        #sample_on_device(location, epoch, args, rate=5, device='cpu')
        models[0].eval()
        with torch.no_grad():
            images =models[0].sample2(4,16)
            save_images(images.detach(), os.path.join("samples", args.run_name, f"epoch_{epoch}_sampling.jpg"))
        models[0].train()





"""
Argument definitions and other options
"""
        
# =============================================================================
# parser = argparse.ArgumentParser()
# args = parser.parse_args()
# 
# import json
# PATH_CFG = 'configs/config.cfg'
# with open(PATH_CFG, 'r') as f:
#     args.__dict__=json.load(f)
# =============================================================================


import json 
PATH_CFG = 'configs/config.cfg'
with open(PATH_CFG, 'r') as f:
    config=Configs(json.load(f))


import glob
if config.resume == True:
    if config.load_backup == False:
        config.resume_path = glob.glob(config.base_path + config.run_name+'/*.pt', recursive=False)[0]
    else:
        config.resume_path = glob.glob(config.base_path + config.backup_name+'/*.pt', recursive=False)[0]


    
# =============================================================================
# input_image = torch.rand((1, 3, 256, 256)).to(torch.device('cuda:0'))
# noise = torch.rand((1, 1, 64, 64)).to(torch.device('cuda:0'))
# 
# vae = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder).cuda()
# 
# output = vae(input_image, noise).cpu().detach()
# plt.imshow(output[0].permute(1, 2, 0))
# =============================================================================





def sample_on_device(location, epoch, args, rate=5, device='cpu'):
    if ((epoch % args.sample_every_n_iter == 0) and (epoch!=0)):
        model = VAE(VAE_Encoder, VAE_Decoder, args)
        model.eval()
        with torch.no_grad():
            ckpt = torch.load(location)
            model.load_state_dict(ckpt['model_state_dict']) #this was missing
            images =model.sample2(24,32)
            save_images(images.detach(), os.path.join("samples", args.run_name, f"{epoch}_orig.jpg"))




import os
from PIL import Image
from torchvision import transforms





# =============================================================================
# pic1=load_image("/home/filipk/Desktop/nature_pics/sea_selection/00000000_(2).jpg")
# 
# save_images(pic1.detach(), "/home/filipk/Desktop/res.jpg")
# 
# =============================================================================



        
        
        
# =============================================================================
#         for i in range(10):
#             images =model.sample2(5,16)
#             epoch='test_noise_3more_readjust'+str(i)
#             save_images(images.detach(), os.path.join("results", args.run_name, f"{epoch}_orig.jpg"))
# =============================================================================
        
# =============================================================================
# import torchvision
# from PIL import Image
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.transforms import v2
# 
# 
# 
# 
# 
# device2 = "cuda"
# model2 = UNet().to(device2)
# ckpt2 = torch.load("/home/filipk/Desktop/Diffusion-Models-pytorch-cutversion/Diffusion-Models-pytorch-main/models/DDPM_Unconditional/ckpt.pt")
# model2.load_state_dict(ckpt2)
# diffusion = Diffusion(img_size=8, device=device)
# x = diffusion.sample(model, n=16)
# =============================================================================













# =============================================================================
# for n in range(10):
#     a=load_image2("/home/filipk/Desktop/Train_latent/Nat_latents2/"+str(n) +".png")
#     
#     
#     
#     device = "cuda"
#     model = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder, args).to(device)
#     model.eval()   
#     with torch.no_grad():
#         ckpt = torch.load(args.resume_path)
#         model.load_state_dict(ckpt['model_state_dict'])
#         img_dec=model.decoder(a.to(torch.device('cuda:0')))
#         save_images(img_dec.detach(),"/home/filipk/Desktop/Train_latent/output/res"+ str(n) +".jpg")
# =============================================================================
# =============================================================================
# from_diffusion=False
# if from_diffusion==True:
# =============================================================================
def decode_from_diffusion(args, model):
    device = "cuda"
    model = VAE(VAE_Encoder, VAE_Decoder, args).to(device)
    model.eval()   
    with torch.no_grad():
        args.resume_path = glob.glob(args.base_path + "VAE_returnto256_square_to16x16"+'/*.pt', recursive=False)[0]
        ckpt = torch.load(args.resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        #x=load_image("/home/filipk/Desktop/TRAIN/archive(2)/00000000_(2).jpg", args)
        #enc=model.encoder(x.to(torch.device('cuda:0')))
        
        
        #Save tensor
        #output_path = "/home/filipk/Desktop/Train_latent/test/"+ "image_name"+'.pt' 
        #torch.save(enc, output_path)
    
        # Load the tensor back
        for i in range(200):
            output_path = "/home/filipk/Desktop/Train_latent/create3/"+ str(i)+'.pt'
            loaded_tensor = torch.load(output_path).unsqueeze(0)
            dec=model.decoder(loaded_tensor.to(torch.device('cuda:0')))
            
        # =============================================================================
        #     enc2=enc.squeeze(0)
        #     
        #     to_pil = transforms.ToPILImage()
        #     pil_image = to_pil(enc2)
        # =============================================================================
                    
        # =============================================================================
        #     # Save the PIL Image to a file
        #     output_path = "/home/filipk/Desktop/Train_latent/test/"+ "image_name"+'.png'  # Change the path and filename as needed
        #     pil_image.save(output_path)
        #     
        #     enc3=load_image2("/home/filipk/Desktop/Train_latent/test/"+ "image_name"+'.png')
        #     
        #     dec=model.decoder(enc3.to(torch.device('cuda:0')))
        # =============================================================================
            
            
            
            save_images(dec.detach(),"/home/filipk/Desktop/Train_latent/output5/res"+ str(i) +".jpg")
    
# =============================================================================
# print("enc:", enc2)
# print("enc3:", enc3.to(torch.device('cuda:0')))
# print("Difference:", torch.sum(torch.abs(enc - enc3.to(torch.device('cuda:0')))))
# 
# =============================================================================


import os
from PIL import Image
from torchvision import transforms
infer =0
if infer ==1:
    device = "cuda"
    model = VAE(VAE_Encoder, VAE_Decoder, args).to(device)
    model.eval()   
    with torch.no_grad():
        ckpt = torch.load(args.resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        
        folder_path = "/home/filipk/Desktop/TRAIN/nature/"

        # Get a list of all files in the folder
        files = os.listdir(folder_path)
        
        # Filter out only image files
        image_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        # Loop through the image files
        for image_file in image_files:
            # Get the full path of the image
            image_path = os.path.join(folder_path, image_file)
        
            # Open the image using Pillow
            #image = Image.open(image_path)
            image_tensor = load_image(image_path,args).to(torch.device('cuda:0'))
            
            # Do something with the image (e.g., display, process, etc.)
            latent = model.encode(image_tensor)
            latent = latent.squeeze(0)
            # Get the name of the image
            image_name = os.path.splitext(image_file)[0]
        
            # Print the image name
            #print("Image Name:", image_name)
            
            # Assuming your image is a torch tensor
            #your_tensor_image = latent  # Replace this with your actual tensor
            
            # Convert the tensor to a PIL Image
# =============================================================================
#             to_pil = transforms.ToPILImage()
#             pil_image = to_pil(your_tensor_image)
#             
# =============================================================================
            # Save the PIL Image to a file
            #output_path = "/home/filipk/Desktop/Train_latent/Nature/"+ image_name+'.png'  # Change the path and filename as needed
            output_path = "/home/filipk/Desktop/Train_latent/Nature_tensors4/"+ image_name+'.pt'  # Change the path and filename as needed
            torch.save(latent, output_path)
            #pil_image.save(output_path)
            
            print(f"Image saved at {output_path}")



def estimate_max_batch_size(input_shape=(3,64,64), device='cuda:1', max_memory_usage=0.9):
    batch_size = 1
    while True:
        try:
            model = VAE(VAE_Encoder, VAE_Decoder, config).to(device)
            inputs = torch.randn((batch_size,) + input_shape).to(device)
            _ = model(inputs)
            batch_size +=1
            print(batch_size)
            del model
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            print(f"Maximum batch size that fits in memory: {batch_size -1 }")
            del model
            del inputs
# =============================================================================
#             del model
#             gc.collect()
#             torch.cuda.empty_cache()
# =============================================================================
            inputs=None
            model=None
            return batch_size -1
            break

            
    gc.collect()
    torch.cuda.empty_cache()   

def test_limit():
    try:
        mem1=estimate_max_batch_size(input_shape=(3,config.image_size,config.image_size), device='cuda:0')
        mem2=estimate_max_batch_size(input_shape=(3,config.image_size,config.image_size), device='cuda:1')
    except:
        print('Something is wrong')
    finally:
        print("gooog")
        print(mem1)
        print(mem2)
        gc.collect()
        torch.cuda.empty_cache()     
    return mem1+mem2
        

            
            
def is_running_in_spyder():
    """Check if the script is running in Spyder or command line."""
    try:
        if 'spyder' in sys.modules:
            return True
        elif 'IPython' in sys.modules and 'spyder' in sys.modules['IPython'].get_ipython().config['KernelApp']['connection_file']:
            return True
    except Exception:
        pass
    return False          


def exclusive_flags(parser, flags):
    """
    Validate that only one of the specified flags is set.
    """
    count = sum(1 for flag in flags if getattr(parser, flag))
    if count != 1:
        raise argparse.ArgumentError(None, "Exactly one of {} must be set.".format(', '.join(flags)))



if __name__ == "__main__":
    if is_running_in_spyder():
        print("Running in Spyder")
        #add way to input the parameters from the window
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
        print("Running from the command line")
        # Create ArgumentParser object
        parser = argparse.ArgumentParser(description='FlexVae model')
        
        # Add positional arguments
        parser.add_argument('-mem', '--flag_mem', action='store_true', help='Check the memmory limits for batch -mem')
        parser.add_argument('-t', '--flag_t', action='store_true', help='Train the model -t')
        parser.add_argument('-i', '--flag_i', action='store_true', help='Infer from the model (full VAE passthrough) -i')
        parser.add_argument('-e', '--flag_e', action='store_true', help='Encode the images in the latent space -e')
        parser.add_argument('opt_pos_arg', type=int, nargs='?', default=None, help='An optional integer positional argument')

        args = parser.parse_args()
        exclusive_flags(args, ['flag_t', 'flag_i', 'flag_e'])
        # Print the argument values
        # Access the flags
        if args.flag_mem:
            print("-mem flag is set")
        if args.flag_t:
            print("-t flag is set")
        if args.flag_i:
            print("-i flag is set")
        if args.flag_e:
            print("-e flag is set")

        print("Argument values:")
        #  print(args.mode)
        print(args.opt_pos_arg)
        
        if args.flag_mem:
           config.batch_size=test_limit()
        if args.flag_t:
            train(config)
        if args.flag_i:
            pass
        if args.flag_e:
            encode_from_folder(config,VAE)
            pass
        
        
            
# =============================================================================
# if __name__ == "__main__":
#     
#     # Create ArgumentParser object
#     parser2 = argparse.ArgumentParser(description='Description of your script')
# 
#     # Add positional arguments
#     parser2.add_argument('pos_arg', type=int, help='A required integer positional argument')
#     parser2.add_argument('opt_pos_arg', type=int, nargs='?', default=None, help='An optional integer positional argument')
# 
#     # Parse the arguments
#     args2 = parser2.parse_args()
# 
#     # Print the argument values
#     print("Argument values:")
#     print(args2.pos_arg)
#     print(args2.opt_pos_arg)
# 
# =============================================================================
            
