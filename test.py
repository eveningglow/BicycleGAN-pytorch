import torch
import torchvision

from dataloader import data_loader
import model
import util

import os
import numpy as np
import argparse

'''
    < make_interpolation >
    Make linear interpolated latent code.
    
    * Parameters
    n : Input images number
    img_num : Generated images number per one input image
    z_dim : Dimension of latent code. Basically 8.
'''
def make_interpolation(n=200, img_num=9, z_dim=8):
    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Make interpolated z
    step = 1 / (img_num-1)
    alpha = torch.from_numpy(np.arange(0, 1, step))
    interpolated_z = torch.FloatTensor(n, img_num, z_dim).type(dtype)

    for i in range(n):
        first_z = torch.randn(1, z_dim)
        last_z = torch.randn(1, z_dim)
        
        for j in range(img_num-1):
            interpolated_z[i, j] = (1 - alpha[j]) * first_z + alpha[j] * last_z
        interpolated_z[i, img_num-1] = last_z
    
    return interpolated_z

'''
    < make_z >
    Make latent code
    
    * Parameters
    n : Input images number
    img_num : Generated images number per one input image
    z_dim : Dimension of latent code. Basically 8.
    sample_type : random or interpolation
'''
def make_z(n, img_num, z_dim=8, sample_type='random'):
    if sample_type == 'random':
        z = util.var(torch.randn(n, img_num, 8))
    elif sample_type == 'interpolation':
        z = util.var(make_interpolation(n=n, img_num=img_num, z_dim=z_dim))
    
    return z


'''
    < make_img >
    Generate images.
    
    * Parameters
    dloader : Dataloader
    G : Generator
    z : Random latent code with size of (N, img_num, z_dim)
    img_size : Image size. Now only 128 is available.
    img_num : Generated images number per one input image.
'''
def make_img(dloader, G, z, img_size=128):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    iter_dloader = iter(dloader)
    img, _ = iter_dloader.next()
    img_num = z.size(1)

    N = img.size(0)    
    img = util.var(img.type(dtype))

    result_img = torch.FloatTensor(N * (img_num + 1), 3, img_size, img_size).type(dtype)

    for i in range(N):
        # The leftmost is domain A image(Edge image)
        result_img[i * (img_num + 1)] = img[i].data

        # Generate img_num images per a domain A image
        for j in range(img_num):
            img_ = img[i].unsqueeze(dim=0)
            z_ = z[i, j, :].unsqueeze(dim=0)
            
            out_img = G(img_, z_)
            result_img[i * (img_num + 1) + j + 1] = out_img.data


    result_img = result_img / 2 + 0.5
    
    return result_img
        
def main(args):    
    dloader, dlen = data_loader(root=args.root, batch_size='all', shuffle=False, 
                                img_size=128, mode='val')

    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    if args.epoch is not None:
        weight_name = '{epoch}-G.pkl'.format(epoch=args.epoch)
    else:
        weight_name = 'G.pkl'
        
    weight_path = os.path.join(args.weight_dir, weight_name)
    G = model.Generator(z_dim=8).type(dtype)
    G.load_state_dict(torch.load(weight_path))
    G.eval()
    
    if os.path.exists(args.result_dir) is False:
        os.makedirs(args.result_dir)
        
    # For example, img_name = random_55.png
    if args.epoch is None:
        args.epoch = 'latest'
    img_name = '{type}_{epoch}.png'.format(type=args.sample_type, epoch=args.epoch)
    img_path = os.path.join(args.result_dir, img_name)

    # Make latent code and images
    z = make_z(n=dlen, img_num=args.img_num, z_dim=8, sample_type=args.sample_type)

    result_img = make_img(dloader, G, z, img_size=128)   
    torchvision.utils.save_image(result_img, img_path, nrow=args.img_num + 1, padding=4)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_type', type=str, choices=['random', 'interpolation'], default='random',
                        help='Type of sampling : \'random\' or \'interpolation\'') 
    parser.add_argument('--root', type=str, default='data/edges2shoes', 
                        help='Data location')
    parser.add_argument('--result_dir', type=str, default='test',
                        help='Ouput images location')
    parser.add_argument('--weight_dir', type=str, default='weight',
                        help='Trained weight location of generator. pkl file location')
    parser.add_argument('--img_num', type=int, default=5,
                        help='Generated images number per one input image')
    parser.add_argument('--epoch', type=int,
                        help='Epoch that you want to see the result. If it is None, the most recent epoch')

    args = parser.parse_args()
    main(args)