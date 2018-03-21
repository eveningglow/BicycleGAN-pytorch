import torch
import argparse
import os

from solver_v6 import Solver

def main(args):
    solver = Solver(root = args.root,
                    result_dir = args.result_dir,
                    weight_dir = args.weight_dir,
                    load_weight = args.load_weight,
                    batch_size = args.batch_size,
                    test_size = args.test_size,
                    test_img_num = args.test_img_num,
                    img_size = args.img_size,
                    num_epoch = args.num_epoch,
                    save_every = args.save_every,
                    lr = args.lr,
                    beta_1 = args.beta_1,
                    beta_2 = args.beta_2,
                    lambda_kl = args.lambda_kl,
                    lambda_img = args.lambda_img,
                    lambda_z = args.lambda_z,
                    z_dim = args.z_dim)
                    
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/edges2shoes', 
                        help='Data location')
    parser.add_argument('--result_dir', type=str, default='test', 
                        help='Result images location')
    parser.add_argument('--weight_dir', type=str, default='weight', 
                        help='Weight location')
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Training batch size')
    parser.add_argument('--test_size', type=int, default=20, 
                        help='Test batch size')
    parser.add_argument('--test_img_num', type=int, default=5, 
                        help='How many images do you want to generate?')
    parser.add_argument('--img_size', type=int, default=128, 
                        help='Image size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5, 
                        help='Beta1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.999, 
                        help='Beta2 for Adam')
    parser.add_argument('--lambda_kl', type=float, default=0.01, 
                        help='Lambda for KL Divergence')
    parser.add_argument('--lambda_img', type=float, default=10, 
                        help='Lambda for image reconstruction')
    parser.add_argument('--lambda_z', type=float, default=0.5, 
                        help='Lambda for z reconstruction')
    parser.add_argument('--z_dim', type=int, default=8, 
                        help='Dimension of z')
    parser.add_argument('--num_epoch', type=int, default=100, 
                        help='Number of epoch')
    parser.add_argument('--save_every', type=int, default=1000, 
                        help='How often do you want to see the result?')
    parser.add_argument('--load_weight', action='store_true',
                        help='Load weight or not')

    args = parser.parse_args()
    main(args)