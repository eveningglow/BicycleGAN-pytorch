# BicycleGAN-pytorch
__Pytorch__ implementation of [BicycleGAN : Toward Multimodal Image-to-Image Translation](https://arxiv.org/abs/1711.11586). This model can generate diverse images from an input image using random latent vector z.

## Result
### Edges2Shoes

## Model description
<p align="center"><img width="100%" src="png/bicyclegan.png" /></p>  

### cVAE-GAN
This can be seen as __image reconstruction process.__ By doing this, we can make the encoder extract proper latent code z which specializes given image 'B' and the generator generate an image which has features of 'B'. Of course, the generator also needs to be able to fool the discriminator. Futhermore, It uses KL-divergence to make the generator be able to generate images using randomly sampled z from normal distribution.

### cLR-GAN
This can be seen as __latent code reconstruction process.__ The main purpose of this process is to make invertible mapping between B and z. It leads to bijective consistency between latent encoding and output modes that is important to prevent from __mode collapse.__ If many latent codes correspond to an output mode, this is mode collapse.

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.2.0](http://pytorch.org/)

## Training step
1. 

## Implementation details

* __Multi discriminator__  
First, two discriminators are used for two different last output size(PatchGAN), 14x14 and 30x30. Second, two discriminators again for images made with encoded z(cVAE-GAN) and random z(cLR-GAN). __Totally, four discriminators are used, (cVAE-GAN, 14x14), (cVAE-GAN, 30x30), (cLR-GAN, 14x14) and (cLR-GAN, 30x30).__

* __Conditional discriminator__  
pass

* __Encoder__  
__E_ResNet__ is used, __not E_CNN__. Residual block in the encoder is slightly different with the usual one. Check ResBlock class and Encoder class in model.py.

### How to inject the latent code z to the generator
Just inject __only to the input__, not to all intermediate layers

### Training data
Batch size is 1 for each cVAE-GAN and cLR-GAN which means that get two images from the dataloader and distribute to cVAE-GAN and cLR-GAN.

### How to encode with encoder
Encoder returns mean and log(variance). Reparameterization trick is used, so __encoded_z = random_z * std + mean__ such that __std = exp(log_variance / 2).__

### How to calculate KL divergence
Images should be here

### How to reconstruct z in cLR-GAN
We get mu and log(variance) as outputs from the encoder in cLR-GAN. Use L1 loss between mu and random_z, not encoded_z and random_z because the latter loss can be unstable if std is big. You can check [here](https://github.com/junyanz/BicycleGAN/issues/14).
