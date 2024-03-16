# Flexible Variational Autoencoder (VAE) 
## To Do:
- [ ] Add more layers to chose from when compiling the structure of encoder and decoder
- [ ] Add latent spaces quantization
- [ ] Add intermidiate layer quantization
- [ ] Add mixed precision support
- [ ] Multi-dim data support
## Working principles
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

