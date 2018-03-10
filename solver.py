import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision

from dataloader import data_loader
import model
import util

import os

'''
    < mse_loss >
    Calculate mean squared error loss

    * Parameters
    score : Output of discriminator
    target : 1 for real and 0 for fake
'''
def mse_loss(score, target=1):
    dtype = type(score)
    
    if target == 1:
        label = util.var(torch.ones(score.size()), requires_grad=False)
    elif target == 0:
        label = util.var(torch.zeros(score.size()), requires_grad=False)
    
    criterion = nn.MSELoss()
    loss = criterion(score, label)
    
    return loss

'''
    < L1_loss >
    Calculate L1 loss

    * Parameters
    pred : Output of network
    target : Ground truth
'''
def L1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

def lr_decay_rule(epoch, start_decay=100, lr_decay=100):
    decay_rate = 1.0 - (max(0, epoch - start_decay) / float(lr_decay))
    return decay_rate

class Solver():
    def __init__(self, root='data/edges2shoes', result_dir='result', weight_dir='weight', load_weight=False,
                 batch_size=2, test_size=20, test_img_num=5, img_size=128, num_epoch=100, save_every=1000,
                 lr=0.0002, beta_1=0.5, beta_2=0.999, lambda_kl=0.01, lambda_img=10, lambda_z=0.5, z_dim=8):
        
        # Data type(Can use GPU or not?)
        self.dtype = torch.cuda.FloatTensor
        if torch.cuda.is_available() is False:
            self.dtype = torch.FloatTensor
        
        # Data loader for training
        self.dloader, dlen = data_loader(root=root, batch_size=batch_size, shuffle=True, 
                                         img_size=img_size, mode='train')

        # Data loader for test
        self.t_dloader, _ = data_loader(root=root, batch_size=test_size, shuffle=False, 
                                        img_size=img_size, mode='val')

        # Models
        # D_cVAE is discriminator for cVAE-GAN(encoded vector z).
        # D_cLR is discriminator for cLR-GAN(random vector z).
        # Both of D_cVAE and D_cLR has two discriminators which have different output size((14x14) and (30x30)).
        # Totally, we have for discriminators now.
        self.D_cVAE = model.Discriminator().type(self.dtype)
        self.D_cLR = model.Discriminator().type(self.dtype)
        self.G = model.Generator(z_dim=z_dim).type(self.dtype)
        self.E = model.Encoder(z_dim=z_dim).type(self.dtype)

        # Optimizers
        self.optim_D_cVAE = optim.Adam(self.D_cVAE.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.optim_D_cLR = optim.Adam(self.D_cLR.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.optim_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.optim_E = optim.Adam(self.E.parameters(), lr=lr, betas=(beta_1, beta_2))

        # fixed random_z for test
        self.fixed_z = util.var(torch.randn(test_size, test_img_num, z_dim))
        
        # Some hyperparameters
        self.z_dim = z_dim
        self.lambda_kl = lambda_kl
        self.lambda_img = lambda_img
        self.lambda_z = lambda_z

        # Extra things
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        self.load_weight = load_weight
        self.test_img_num = test_img_num
        self.img_size = img_size
        self.start_epoch = 0
        self.num_epoch = num_epoch
        self.save_every = save_every
        
    '''
        < show_model >
        Print model architectures
    '''
    def show_model(self):
        print('=========================== Discriminator for cVAE ===========================')
        print(self.D_cVAE)
        print('=============================================================================\n\n')
        print('=========================== Discriminator for cLR ===========================')
        print(self.D_cLR)
        print('=============================================================================\n\n')
        print('================================= Generator =================================')
        print(self.G)
        print('=============================================================================\n\n')
        print('================================== Encoder ==================================')
        print(self.E)
        print('=============================================================================\n\n')
        
    '''
        < set_train_phase >
        Set training phase
    '''
    def set_train_phase(self):
        self.D_cVAE.train()
        self.D_cLR.train()
        self.G.train()
        self.E.train()
        
    '''
        < load_pretrained >
        If you want to continue to train, load pretrained weight
    '''
    def load_pretrained(self):
        self.D_cVAE.load_state_dict(torch.load(os.path.join(self.weight_dir, 'D_cVAE.pkl')))
        self.D_cLR.load_state_dict(torch.load(os.path.join(self.weight_dir, 'D_cLR.pkl')))
        self.G.load_state_dict(torch.load(os.path.join(self.weight_dir, 'G.pkl')))
        self.E.load_state_dict(torch.load(os.path.join(self.weight_dir, 'E.pkl')))
        
        log_file = open('log.txt', 'r')
        line = log_file.readline()
        self.start_epoch = int(line)
        
    '''
        < save_weight >
        Save weight
    '''
    def save_weight(self, epoch=None):
        if epoch is None:
            d_cVAE_name = 'D_cVAE.pkl'
            d_cLR_name = 'D_cLR.pkl'
            g_name = 'G.pkl'
            e_name = 'E.pkl'
        else:
            d_cVAE_name = '{epochs}-{name}'.format(epochs=str(epoch), name='D_cVAE.pkl')
            d_cLR_name = '{epochs}-{name}'.format(epochs=str(epoch), name='D_cLR.pkl')
            g_name = '{epochs}-{name}'.format(epochs=str(epoch), name='G.pkl')
            e_name = '{epochs}-{name}'.format(epochs=str(epoch), name='E.pkl')
            
        torch.save(self.D_cVAE.state_dict(), os.path.join(self.weight_dir, d_cVAE_name))
        torch.save(self.D_cVAE.state_dict(), os.path.join(self.weight_dir, d_cLR_name))
        torch.save(self.G.state_dict(), os.path.join(self.weight_dir, g_name))
        torch.save(self.E.state_dict(), os.path.join(self.weight_dir, e_name))
    
    '''
        < all_zero_grad >
        Set all optimizers' grad to zero 
    '''
    def all_zero_grad(self):
        self.optim_D_cVAE.zero_grad()
        self.optim_D_cLR.zero_grad()
        self.optim_G.zero_grad()
        self.optim_E.zero_grad()
        
    '''
        < train >
        Train the D_cVAE, D_cLR, G and E 
    '''
    def train(self):
        if self.load_weight is True:
            self.load_pretrained()
        
        self.set_train_phase()
        self.show_model()
        
        # Training Start!
        for epoch in range(self.start_epoch, self.num_epoch):
            for iters, (img, ground_truth) in enumerate(self.dloader):
                # img(2, 3, 128, 128) : Domain A. One for cVAE and another for cLR. 
                # ground_truth(2, 3, 128, 128) : Domain B. One for cVAE and another for cLR.
                img, ground_truth = util.var(img), util.var(ground_truth)

                # Seperate data for cVAE_GAN(using encoded z) and cLR_GAN(using random z)
                cVAE_data = {'img' : img[0].unsqueeze(dim=0), 'ground_truth' : ground_truth[0].unsqueeze(dim=0)}
                cLR_data = {'img' : img[1].unsqueeze(dim=0), 'ground_truth' : ground_truth[1].unsqueeze(dim=0)}

                ''' ----------------------------- 1. Train D ----------------------------- '''
                #############   Step 1. D loss in cVAE-GAN(See Figure 2.(c))   #############

                # Encoded latent vector
                mu, log_variance = self.E(cVAE_data['ground_truth'])
                std = torch.exp(log_variance / 2)
                random_z = util.var(torch.randn(1, self.z_dim))
                encoded_z = (random_z * std) + mu

                # Generate fake image
                fake_img_cVAE = self.G(cVAE_data['img'], encoded_z)

                # Get scores and loss
                real_d_cVAE_1, real_d_cVAE_2 = self.D_cVAE(cVAE_data['ground_truth'])
                fake_d_cVAE_1, fake_d_cVAE_2 = self.D_cVAE(fake_img_cVAE)
                
                # mse_loss for LSGAN
                D_loss_cVAE_1 = mse_loss(real_d_cVAE_1, 1) + mse_loss(fake_d_cVAE_1, 0)
                D_loss_cVAE_2 = mse_loss(real_d_cVAE_2, 1) + mse_loss(fake_d_cVAE_2, 0)
                
                #############   Step 2. D loss in cLR-GAN(See Figure 2.(D))   #############

                # Random latent vector
                random_z = util.var(torch.randn(1, self.z_dim))

                # Generate fake image
                fake_img_cLR = self.G(cLR_data['img'], random_z)

                # Get scores and loss
                # Big PatchGAN Discriminator
                real_d_cLR_1, real_d_cLR_2 = self.D_cLR(cLR_data['ground_truth'])
                fake_d_cLR_1, fake_d_cLR_2 = self.D_cLR(fake_img_cLR)
                
                D_loss_cLR_1 = mse_loss(real_d_cLR_1, 1) + mse_loss(fake_d_cLR_1, 0)
                D_loss_cLR_2 = mse_loss(real_d_cLR_2, 1) + mse_loss(fake_d_cLR_2, 0)

                D_loss = D_loss_cVAE_1 + D_loss_cLR_1 + D_loss_cVAE_2 + D_loss_cLR_2

                # Update
                self.all_zero_grad()
                D_loss.backward()
                self.optim_D_cVAE.step()
                self.optim_D_cLR.step()

                ''' ----------------------------- 2. Train G & E ----------------------------- '''
                # Step 1. GAN loss to fool discriminator (cVAE_GAN and cLR_GAN)

                # Encoded latent vector
                mu, log_variance = self.E(cVAE_data['ground_truth'])
                std = torch.exp(log_variance / 2)
                random_z = util.var(torch.randn(1, self.z_dim))
                encoded_z = (random_z * std) + mu

                # Generate fake image and get adversarial loss
                fake_img_cVAE = self.G(cVAE_data['img'], encoded_z)
                fake_d_cVAE_1, fake_d_cVAE_2 = self.D_cVAE(fake_img_cVAE)

                GAN_loss_cVAE_1 = mse_loss(fake_d_cVAE_1, 1)
                GAN_loss_cVAE_2 = mse_loss(fake_d_cVAE_2, 1)

                # Random latent vector
                random_z = util.var(torch.randn(1, self.z_dim))

                # Generate fake image and get adversarial loss
                fake_img_cLR = self.G(cLR_data['img'], random_z)
                fake_d_cLR_1, fake_d_cLR_2 = self.D_cLR(fake_img_cLR)

                GAN_loss_cLR_1 = mse_loss(fake_d_cLR_1, 1)
                GAN_loss_cLR_2 = mse_loss(fake_d_cLR_2, 1)

                G_GAN_loss = GAN_loss_cVAE_1 + GAN_loss_cVAE_2 + GAN_loss_cLR_1 + GAN_loss_cLR_2

                # Step 2. KL-divergence with N(0, 1) (cVAE-GAN)
                # See http://yunjey47.tistory.com/43 or Appendix B in the paper for details
                KL_div = self.lambda_kl * torch.sum(0.5 * (mu ** 2 + torch.exp(log_variance) - log_variance - 1))

                # Step 3. Reconstruction of ground truth image (|G(A, z) - B|) (cVAE-GAN)
                img_recon_loss = self.lambda_img * L1_loss(fake_img_cVAE, cVAE_data['ground_truth'])

                EG_loss = G_GAN_loss + KL_div + img_recon_loss
                self.all_zero_grad()
                EG_loss.backward(retain_graph=True)
                self.optim_E.step()
                self.optim_G.step()

                ''' ----------------------------- 3. Train ONLY G ----------------------------- '''
                # Step 1. Reconstrution of random latent code (|E(G(A, z)) - z|) (cLR-GAN)
                # This step should update only G.
                # See https://github.com/junyanz/BicycleGAN/issues/5 for details.
                mu_, log_variance_ = self.E(fake_img_cLR)
                z_recon_loss = L1_loss(mu_, random_z)

                G_alone_loss = self.lambda_z * z_recon_loss

                self.all_zero_grad()
                G_alone_loss.backward()
                self.optim_G.step()

                log_file = open('log.txt', 'w')
                log_file.write(str(epoch))
                
                # Print error, save intermediate result image and weight
                if iters % self.save_every == 0:
                    print('[Epoch : %d / Iters : %d] => D_loss : %f / G_GAN_loss : %f / KL_div : %f / img_recon_loss : %f / z_recon_loss : %f'\
                          %(epoch, iters, D_loss.data[0], G_GAN_loss.data[0], KL_div.data[0], img_recon_loss.data[0], G_alone_loss.data[0]))

                    # Save intermediate result image
                    if os.path.exists(self.result_dir) is False:
                        os.makedirs(self.result_dir)

                    result_img = util.make_img(self.t_dloader, self.G, self.fixed_z, 
                                               img_num=self.test_img_num, img_size=self.img_size)

                    img_name = '{epoch}_{iters}.png'.format(epoch=epoch, iters=iters)
                    img_path = os.path.join(self.result_dir, img_name)

                    torchvision.utils.save_image(result_img, img_path, nrow=self.test_img_num+1)

                    # Save intermediate weight
                    if os.path.exists(self.weight_dir) is False:
                        os.makedirs(self.weight_dir)
                    
                    self.save_weight()
                    
            # Save weight at the end of every epoch
            self.save_weight(epoch=epoch)
