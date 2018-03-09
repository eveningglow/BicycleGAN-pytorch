# BicycleGAN-pytorch
__Pytorch__ implementation of [BicycleGAN : Toward Multimodal Image-to-Image Translation](https://arxiv.org/abs/1711.11586). This model can generate diverse images from an input image using random latent vector z.
## Result
### Edges2Shoes

## Model description
### cVAE-GAN

### cLR-GAN

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.2.0](http://pytorch.org/)

## Training step

## Implementation details
### Details in network structure
#### * Multi discriminator
First, two discriminators are used for two different last output size(PatchGAN), 14x14 and 30x30. Second, two discriminators again for images made with encoded z(cVAE-GAN) and random z(cLR-GAN). __Totally, four discriminators are used, (cVAE-GAN, 14x14), (cVAE-GAN, 30x30), (cLR-GAN, 14x14) and (cLR-GAN, 30x30).__

#### * Conditional discriminator
pass

#### * Encoder
__E_ResNet__ is used, __not E_CNN__. Residual block in the encoder is slightly different with the usual one. Check ResBlock class and Encoder class in model.py.

#### * How to inject the latent code z to the generator
Just inject __only to the input__, not to all intermediate layers

### Details in training process
#### 1. Training data
Batch size is 1 for each cVAE-GAN and cLR-GAN which means that get two images from the dataloader and distribute to cVAE-GAN and cLR-GAN.

#### 2. How to encode with encoder
Encoder returns mean and log(variance). Reparameterization trick is used, so __encoded_z = random_z * std + mean__ such that __std = exp(log_variance / 2).__

#### 3. 
