# Flexible Variational Autoencoder (VAE) 
## To Do:
- [ ] Add more layers to chose from when compiling the structure of encoder and decoder
- [ ] Add latent spaces quantization
- [ ] Add intermidiate layer quantization
- [ ] Add mixed precision support
- [ ] Multi-dim data support
## Working principles
## Implementation choices
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
  "C2d_8_8_1_1_0",
  "MaxP_2_2"
]
```

![Example structure of VAE encoder](/assets/encod.png)
## How to use
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$
