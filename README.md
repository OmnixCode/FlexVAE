# Flexible Variational Autoencoder (VAE) 
## To Do:
- [ ] Add more layers to chose from when compiling the structure of encoder and decoder
- [ ] Add latent spaces quantization
- [ ] Add intermidiate layer quantization
- [ ] Add mixed precision support
- [ ] Multi-dim data support
## Working principles

### Autoencoders
An autoencoder is a type of artificial neural network used for unsupervised learning of efficient data codings. 
The network aims to learn a compressed representation (encoding) of the input data, often referred to as the latent space, 
and then reconstruct the original input from this representation (decoding).

Here's a basic description of the autoencoder network:

Encoder: The encoder part of the network takes the input data and maps it to a lower-dimensional latent space representation. 
It typically consists of one or more layers of neurons that progressively reduce the dimensionality of the input data. 
The final layer of the encoder produces the latent representation.

Latent Space: The latent space representation is a compressed form of the input data, capturing its essential features. 
It's essentially the bottleneck of the autoencoder, where the network aims to learn a compact and meaningful representation.

Decoder: The decoder part of the network takes the latent representation and reconstructs the original input data from it. 
Similar to the encoder, the decoder consists of one or more layers of neurons that progressively upsample 
the latent representation to match the dimensions of the original input.

Loss Function: The training objective of the autoencoder is typically to minimize the reconstruction error between the input data and the output of the decoder. 
Common loss functions used for this purpose include mean squared error (MSE) or binary cross-entropy loss, depending on the nature of the input data.

Training: The autoencoder is trained using backpropagation and gradient descent-based optimization algorithms. 
During training, the encoder and decoder parameters are updated iteratively to minimize the reconstruction error on a training dataset.

Autoencoders are capable of learning efficient representations of data and can be used for various tasks such as data denoising, 
dimensionality reduction, and generative modeling. Variants of autoencoders include denoising autoencoders, sparse autoencoders, 
and variational autoencoders, each tailored for specific applications or constraints.


### Variational Autoencoders
Compared to regular autoencoders, variational autoencders introduce additional loss term that forces the latent space to have standard normal distribution (normal distribution wiht zero mean and variance of 1).
Instead of the direct passing of the latent space variable from the encoder to the decoder, we first calculate mean and the variance of this variable. Afterwards, we randomly sample latent tensor (vector) from the distribution with this mean and variance and pass it to the decoder. To estimate how different is this distribution to the normal distribution we use [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).

## Implementation choices
### Model definition
Encoder and decoder networks used in VAE are defined with the help of structure JSON files.
This list is then passed to sequential model generator.
In order to shorten notation, we can also string multiple same layers with the help of multiplication operator.
i.e. "2*VAERB_256_256" is short notation for:
```json
[
  "VAERB_256_256",
  "VAERB_256_256",
]
```
Parameters are passed after the layer abbreviation
i.e. "C2d_3_128_3_1_1" is a Conv2D layer with:
```
in_channels = 3, 
out_channels = 128, 
kernel_size = 3, 
stride=1, 
padding=1
```

Available layers and their abbreviations:
```python
{
'C2d' : nn.Conv2d, #layer from torch.nn
'VAERB' : VAE_ResidualBlock, #defined in src/modules
'VAEA' : VAE_AttentionBlock, #defined in src/modules
'GN' : nn.GroupNorm, #layer from torch.nn
'SiLU' : nn.SiLU, #layer from torch.nn
'MaxP' : nn.MaxPool2d, #layer from torch.nn
'UpS' : nn.Upsample #layer from torch.nn
}
                
```




Our basic encoder model is defined as:

```json
[
  "C2d_3_128_3_1_1",
  "2*VAERB_128_128",
  "C2d_128_128_3_4_0",
  "VAERB_128_256",
  "VAERB_256_256",
  "C2d_256_256_3_4_0",
  "3*VAERB_256_256",
  "VAEA_256",
  "VAERB_256_256",
  "GN_32_256",
  "SiLU",
  "C2d_256_8_3_1_1",
  "C2d_8_8_1_1_0"
]
```
We can also visualise its structure:

![Example structure of VAE encoder](/assets/encod.png)

Alternatively basic decoder model is defined as:
```json
[
  "C2d_4_4_1_1_0",
  "C2d_4_512_3_1_1",
  "VAERB_512_512",
  "VAEA_512",
  "4*VAERB_512_512",
  "UpS_4",
  "C2d_512_512_3_1_1",
  "VAERB_512_256",
  "2*VAERB_256_256",
  "UpS_4",
  "C2d_256_256_3_1_1",
  "VAERB_256_128",
  "2*VAERB_128_128",
  "GN_32_128",
  "SiLU",
  "C2d_128_3_3_1_1"
]
```

### Loss functions

Loss function for the the model is expressed as: 
$$L_{total} = \alpha  \cdot  L_{reconstruction} + kld{\textunderscore}weight  \cdot  L_{KLD} + ssim{\textunderscore}metrics  \cdot  \beta  \cdot  (1-L_{ssim}) + sparse{\textunderscore}metrics \cdot L\_{sparse} $$

Where $ssim{\textunderscore}metrics$ and $sparse{\textunderscore}metrics$ are logical variables that take the ${\color{red}True}$ or ${\color{red}False}$ values.
This is internaly converted to 0 or 1 and this effectively turns off or on the SSIM and sparse parts of the total loss.


## How to use

### Basic modes
Model configuration file is present in /configs/config.cfg file. Here we can adjust all the input, output and training parameters.
To run the model run the train.py script as:
```
python3 ./train.py [mode]
```

Available modes are:<br />
**-h**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;help <br />
**-t**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train <br />
**-i**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;inference (full pass through the VAE network (encoder + decoder)) <br />
**-e**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;encode (encode the images from the folder into latent representation) <br />
**-mem** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; estimate the maximum batch size for training  <br />
### Inference example

<p align="center">
    <img width="800" src="/assets/nature.jpg" alt="VAE reconstruction">
    <br> <!-- Optional line break for better separation -->
    Original images of the natural scenery to be fed to the network <!-- Description for the image -->
</p>


<p align="center">
    <img width="800" src="/assets/nature_reconstructed.jpg" alt="VAE reconstruction">
    <br> <!-- Optional line break for better separation -->
    Reconstructed image <!-- Description for the image -->
</p>



