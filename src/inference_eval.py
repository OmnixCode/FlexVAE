#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import os
from utils import load_image, save_images
from modules import  VAE_Encoder, VAE_Decoder
import glob


def encode_from_folder(config, model):
    '''
    Function for encoding all the images in all the folders in some defined folder. 
    This is useful for further training of diffusion networks, or analysis of the latent space.
    Latent images are saved as a tensor in .pt format.

    Parameters
    ----------
    config : config struct defined in utils.py.

    Returns
    -------
    None.

    '''
    
    device = "cuda"
    #model = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder, args).to(device)
    model = model(VAE_Encoder, VAE_Decoder, config).to(device)
    model.eval()   
    with torch.no_grad():
        ckpt = torch.load(config.resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        
        #folder_path = "/home/filipk/Desktop/TRAIN/nature/"
        
        #folder_path_base = "/home/filipk/Desktop/TRAIN/"
        folder_path_base = config.encode_path
        paths=[x[1] for x in os.walk(folder_path_base)]
        #save_PATH = "/home/filipk/Desktop/Train_latent/Nature_tensors3/"
        save_PATH = config.encode_save_path
        
        for path in paths[0]:
            folder_path=folder_path_base+path
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
                image_tensor = load_image(image_path,config).to(torch.device('cuda:0'))
                
                # Do something with the image (e.g., display, process, etc.)
                latent = model.encode(image_tensor)
                latent = latent.squeeze(0)
                # Get the name of the image
                image_name = os.path.splitext(image_file)[0]
            
                os.makedirs(save_PATH+path, exist_ok=True)
                output_path = save_PATH+path+"/"+ image_name+'.pt'  # Change the path and filename as needed
                torch.save(latent, output_path)
                #pil_image.save(output_path)
                
                print(f"Image saved at {output_path}")
                
                
def infer_from_folder(config, model):
    '''
    Function for inference on all the images in all the folders in some defined folder. 

    Parameters
    ----------
    config : config struct defined in utils.py.

    Returns
    -------
    None.

    '''
    
    device = "cuda"
    #model = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder, args).to(device)
    model = model(VAE_Encoder, VAE_Decoder, config).to(device)
    model.eval()   
    with torch.no_grad():
        ckpt = torch.load(config.resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        
        #folder_path = "/home/filipk/Desktop/TRAIN/nature/"
        
        #folder_path_base = "/home/filipk/Desktop/TRAIN/"
        folder_path_base = config.infer_folder_input
        paths=[x[1] for x in os.walk(folder_path_base)]
        #save_PATH = "/home/filipk/Desktop/Train_latent/Nature_tensors3/"
        save_PATH = config.infer_folder_output
        
        for path in paths[0]:
            folder_path=folder_path_base+path
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
                image_tensor = load_image(image_path,config).to(torch.device('cuda:0'))
                
                # Do something with the image (e.g., display, process, etc.)
                latent =model(image_tensor)
                
                # Get the name of the image
                image_name = os.path.splitext(image_file)[0]
                          
                os.makedirs(save_PATH+path, exist_ok=True)
                output_path = save_PATH+path+"/"+ image_name+'.jpg'  # Change the path and filename as needed
               # torch.save(latent, output_path)
                save_images(latent .detach(), output_path)
                #pil_image.save(output_path)
                
                print(f"Image saved at {output_path}")
                
                
                
                
def decode_from_diffusion(config, model):
    
    '''
    Function for decoding images produced by the encoder (or some other network) in one folder. 

    Parameters
    ----------
    config : config struct defined in utils.py.

    Returns
    -------
    None.

    '''
    
    device = "cuda"
    model = model(VAE_Encoder, VAE_Decoder, config).to(device)
    model.eval()   
    with torch.no_grad():
        config.resume_path = glob.glob(config.base_path + "VAE_returnto256_square_to16x16"+'/*.pt', recursive=False)[0]
        ckpt = torch.load(config.resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        #base_path="/home/filipk/Desktop/Train_latent/create3/"
        base_path=config.decode_path
        save_path=config.decode_save_path
        #save_folder= "/home/filipk/Desktop/Train_latent/output5/res"
        # Load the tensor back
        for i in range(200):
            input_path = base_path + str(i)+'.pt'
            loaded_tensor = torch.load(input_path).unsqueeze(0)
            dec=model.decoder(loaded_tensor.to(torch.device('cuda:0')))
            
            save_images(dec.detach(), save_path+'res'+ str(i) +".jpg")
            
            
def interpolate(img1, img2, percentage, config, model):
    '''
    Function for interpolating between the two images in the latent space.

    Parameters
    ----------
    img1 : TYPE
        DESCRIPTION.
    img2 : TYPE
        DESCRIPTION.
    percentage : float between 0 and 1.
    config : config struct defined in utils.py.
    model : TYPE
        DESCRIPTION.

    Raises
    ------
    the
        if interpolation is not between 0 and 1 output error
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    device = "cuda"
    model = model(VAE_Encoder, VAE_Decoder, config).to(device)
    model.eval()   #put model into evaluation mode (batch norms and other things are fixed)
    with torch.no_grad():
        #script_module = torch.jit.load("/home/filipk/Desktop/VAE_SD_LOWRES/models/VAE/ckpt.pt")
        #ckpt = torch.load("/home/filipk/Desktop/VAE_FINAL_v2/models/VAE_returnto128_nodrop/128_to_256_kld_mult_0.01_epoch_546ckpt.pt")
        ckpt = torch.load(config.resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        
        load_folder1=config.interpolate_folder1
        load_folder2=config.interpolate_folder2
        interpolation_output_folder= config.inter_out
        #images used for interpolation
        pic1=load_image(load_folder1+str(img1)).to(torch.device('cuda:0'))
        pic2=load_image(load_folder2+str(img2)).to(torch.device('cuda:0'))
        if (percentage < 0) or (percentage > 1):   
            # raise the ValueError
            raise ValueError("Please add a value for percentage between 0 and 1")
        #pic_in= (pic1*percentage+pic2*(1-percentage))/2
        pic_in= (pic1*percentage+pic2*(1-percentage))/2
        
        #noise = torch.randn((pic1.size(0), 4, 256)).to(torch.device('cuda:0'))
        #pic_res=model(pic_in,noise)
        pic_res=model(pic_in)
        save_images(pic_res.detach(), interpolation_output_folder + "res.jpg")