#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:58:59 2024

@author: filipk
"""

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
    config : TYPE
        DESCRIPTION.

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
    config : TYPE
        DESCRIPTION.

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
                #latent = model.encode(image_tensor)
                #image_infer = model.decoder(latent)
                #latent = image_infer.squeeze(0)
                
                latent =model(image_tensor)
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
    config : TYPE
        DESCRIPTION.

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