# BicycleGAN-pytorch
__Pytorch__ implementation of [BicycleGAN : Toward Multimodal Image-to-Image Translation](https://arxiv.org/abs/1711.11586).
<p align="center"><img width="100%" src="png/represent.png" /></p>  

## Result
### Edges2Shoes
* Random sampling
<p align="center"><img width="100%" src="png/random_sample.png" /></p>  

* Linear interpolated sampling
<p align="center"><img width="100%" src="png/interpolation.png" /></p>  

## Model description
<p align="center"><img width="100%" src="png/model.png" /></p>  

### cVAE-GAN
This can be seen as __image reconstruction process.__ By doing this, we can make the encoder extract proper latent code z which contains features of given image 'B' and the generator generate image which has features of 'B'. Of course, the generator also needs to be able to fool the discriminator. Futhermore, It uses KL-divergence to make the generator be able to generate images using randomly sampled z from normal distribution at the test phase.

### cLR-GAN
This can be seen as __latent code reconstruction process.__ If many latent codes correspond to a same output mode, this is mode collapse. The main purpose of cLR-GAN is to make invertible mapping between B and z. It leads to bijective consistency between latent encoding and output modes that is significant to prevent model from __mode collapse.__  

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.2.0](http://pytorch.org/)

## Training step  
Before getting started, suppose that we want to optmize G which can convert __domain A into B__.  
__real_B__ : A real image of domain B from training data set  
__fake_B__ : A fake image of domain B made by the generator  
  
  
__1. Optimize D__ 
* Optimize D in cVAE-GAN using real_B and fake_B made with encoded_z(__Adversarial loss__).  
* Optimize D in cLR-GAN using real_B and fake_B made with random_z(__Adversarial loss__).  

__2. Optimize G or E__  
* Optimize G and E in cVAE-GAN using fake_B made with encoded_z(__Adversarial loss__).
* Optimize G and E in cVAE-GAN using real_B and fake_B made with encoded_z(__Image reconstruction loss__).  
* Optimize E in cVAE-GAN using the encoder outputs, mu and log_variance(__KL-div loss__).  
* Optimize G in cLR-GAN using fake_B made with random_z(__Adversarial loss__).  

__3. Optimize ONLY G(Do not update E)__  
* Optimize G in cLR-GAN using random_z and the encoder output mu(__Latent code reconstruction loss__).

## Implementation details

* __Multi discriminator__  
First, two discriminators are used for two different last output size(PatchGAN), 14x14 and 30x30. Second, two discriminators again for images made with encoded z(cVAE-GAN) and random z(cLR-GAN). Totally, __four discriminators__ are used, __(cVAE-GAN, 14x14), (cVAE-GAN, 30x30), (cLR-GAN, 14x14) and (cLR-GAN, 30x30).__

* __Encoder__  
__E_ResNet__ is used, __not E_CNN__. Residual block in the encoder is slightly different with the usual one. Check ResBlock class and Encoder class in model.py.

* __How to inject the latent code z to the generator__  
Just inject __only to the input__, not to all intermediate layers

* __Training data__  
Batch size is 1 for each cVAE-GAN and cLR-GAN which means that get two images from the dataloader and distribute to cVAE-GAN and cLR-GAN.

* __How to encode with encoder__  
Encoder returns mean and log(variance). Reparameterization trick is used, so __encoded_z = random_z * std + mean__ such that __std = exp(log_variance / 2).__

* __How to calculate KL divergence__  
<p align="left"><img width="70%" img height="70%" src="png/kl_1.png" /></p>  
We need to get KL divergence with N(0, 1), so it leads to following expression.  
<p align="left"><img width="70%" img height="70%" src="png/kl_2.png" /></p>  

* __How to reconstruct z in cLR-GAN__  
We get mu and log(variance) as outputs from the encoder in cLR-GAN. Use __L1 loss between mu and random_z__, not encoded_z and random_z because the latter loss can be unstable if std is big. You can check [here](https://github.com/junyanz/BicycleGAN/issues/14).

## How to train
```python train.py --root=data/edges2shoes --result_dir=result --weight_dir=weight```
