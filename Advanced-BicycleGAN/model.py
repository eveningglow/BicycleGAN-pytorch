import torch
import torch.nn as nn

'''
    < ConvBlock >
    Small unit block consists of [convolution layer - normalization layer - non linearity layer]
    
    * Parameters
    1. in_dim : Input dimension(channels number)
    2. out_dim : Output dimension(channels number)
    3. k : Kernel size(filter size)
    4. s : stride
    5. p : padding size
    6. norm : If it is true add Instance Normalization layer, otherwise skip this layer
    7. non_linear : You can choose between 'leaky_relu', 'relu', 'None'
'''
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []
        
        # Convolution Layer
        layers += [nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]
        
        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]
            
        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        
        self.conv_block = nn.Sequential(* layers)
        
    def forward(self, x):
        out = self.conv_block(x)
        return out
    
'''
    < DeonvBlock >
    Small unit block consists of [transpose conv layer - normalization layer - non linearity layer]
    
    * Parameters
    1. in_dim : Input dimension(channels number)
    2. out_dim : Output dimension(channels number)
    3. k : Kernel size(filter size)
    4. s : stride
    5. p : padding size
    6. norm : If it is true add Instance Normalization layer, otherwise skip this layer
    7. non_linear : You can choose between 'relu', 'tanh', None
'''
class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='relu'):
        super(DeconvBlock, self).__init__()
        layers = []
        
        # Transpose Convolution Layer
        layers += [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]
        
        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]
        
        # Non-Linearity Layer
        if non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif non_linear == 'tanh':
            layers += [nn.Tanh()]
            
        self.deconv_block = nn.Sequential(* layers)
            
    def forward(self, x):
        out = self.deconv_block(x)
        return out

'''
    < Generator >
    U-Net Generator. See https://arxiv.org/abs/1505.04597 figure 1 
    or https://arxiv.org/pdf/1611.07004 6.1.1 Generator Architectures
    
    Downsampled activation volume and upsampled activation volume which have same width and height
    make pairs and they are concatenated when upsampling.
    Pairs : (up_1, down_6) (up_2, down_5) (up_3, down_4) (up_4, down_3) (up_5, down_2) (up_6, down_1)
            down_7 doesn't have a partener.
    
    ex) up_1 and down_6 have same size of (N, 512, 2, 2) given that input size is (N, 3, 128, 128).
        When forwarding into upsample_2, up_1 and down_6 are concatenated to make (N, 1024, 2, 2) and then
        upsample_2 makes (N, 512, 4, 4). That is why upsample_2 has 1024 input dimension and 512 output dimension 
        
        Except upsample_1, all the other upsampling blocks do the same thing.
'''
class Generator(nn.Module):
    def __init__(self, z_dim=8): 
        super(Generator, self).__init__()
        # Reduce H and W by half at every downsampling
        self.downsample_1 = ConvBlock(3 + z_dim, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu')
        self.downsample_2 = ConvBlock(64, 128, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_3 = ConvBlock(128, 256, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_4 = ConvBlock(256, 512, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_5 = ConvBlock(512, 512, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_6 = ConvBlock(512, 512, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_7 = ConvBlock(512, 512, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        
        # Need concatenation when upsampling, see foward function for details
        self.upsample_1 = DeconvBlock(512, 512, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.upsample_2 = DeconvBlock(1024, 512, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.upsample_3 = DeconvBlock(1024, 512, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.upsample_4 = DeconvBlock(1024, 256, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.upsample_5 = DeconvBlock(512, 128, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.upsample_6 = DeconvBlock(256, 64, k=4, s=2, p=1, norm=True, non_linear='relu')
        self.upsample_7 = DeconvBlock(128, 3, k=4, s=2, p=1, norm=False, non_linear='Tanh')
    
    def forward(self, x, z):
        # z : (N, z_dim) -> (N, z_dim, 1, 1) -> (N, z_dim, H, W)
        # x_with_z : (N, 3 + z_dim, H, W)
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat([x, z], dim=1)
        
        down_1 = self.downsample_1(x_with_z)
        down_2 = self.downsample_2(down_1)
        down_3 = self.downsample_3(down_2)
        down_4 = self.downsample_4(down_3)
        down_5 = self.downsample_5(down_4)
        down_6 = self.downsample_6(down_5)
        down_7 = self.downsample_7(down_6)

        up_1 = self.upsample_1(down_7)
        up_2 = self.upsample_2(torch.cat([up_1, down_6], dim=1))
        up_3 = self.upsample_3(torch.cat([up_2, down_5], dim=1))
        up_4 = self.upsample_4(torch.cat([up_3, down_4], dim=1))
        up_5 = self.upsample_5(torch.cat([up_4, down_3], dim=1))
        up_6 = self.upsample_6(torch.cat([up_5, down_2], dim=1))
        out = self.upsample_7(torch.cat([up_6, down_1], dim=1))
        
        return out 
    
'''
    < Discriminator >
    
    PatchGAN discriminator. See https://arxiv.org/pdf/1611.07004 6.1.2 Discriminator architectures.
    It uses two discriminator which have different output sizes(different local probabilities).
    
    Futhermore, it is conditional discriminator so input dimension is 6. You can make input by concatenating
    two images to make pair of Domain A image and Domain B image. 
    There are two cases to concatenate, [Domain_A, Domain_B_ground_truth] and [Domain_A, Domain_B_generated]
    
    d_1 : (N, 6, 128, 128) -> (N, 1, 14, 14)
    d_2 : (N, 6, 128, 128) -> (N, 1, 30, 30)
    
    In training, the generator needs to fool both of d_1 and d_2 and it makes the generator more robust.
 
'''  
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()       
        # Discriminator with last patch (14x14)
        # (N, 6, 128, 128) -> (N, 1, 14, 14)
        self.d_1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=0, count_include_pad=False),
                                 ConvBlock(6, 32, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(32, 64, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(64, 128, k=4, s=1, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 1, k=4, s=1, p=1, norm=False, non_linear=None))
        
        # Discriminator with last patch (30x30)
        # (N, 6, 128, 128) -> (N, 1, 30, 30)
        self.d_2 = nn.Sequential(ConvBlock(6, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(64, 128, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 256, k=4, s=1, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(256, 1, k=4, s=1, p=1, norm=False, non_linear=None))
    
    def forward(self, x):
        out_1 = self.d_1(x)
        out_2 = self.d_2(x)
        return (out_1, out_2)
    
'''
    < ResBlock >
    
    This residual block is different with the one we usaully know which consists of 
    [conv - norm - act - conv - norm] and identity mapping(x -> x) for shortcut.
    
    Also spatial size is decreased by half because of AvgPool2d.
'''
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        
        self.short_cut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out
        
'''
    < Encoder >
    
    Output is mu and log(var) for reparameterization trick used in Variation Auto Encoder.
    Encoding is done in this order.
    1. Use this encoder and get mu and log_var
    2. std = exp(log(var / 2))
    3. random_z = N(0, 1)
    4. encoded_z = random_z * std + mu (Reparameterization trick)
'''
class Encoder(nn.Module):
    def __init__(self, z_dim=8):
        super(Encoder, self).__init__()
        
        self.conv = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.res_blocks = nn.Sequential(ResBlock(64, 128),
                                        ResBlock(128, 192),
                                        ResBlock(192, 256))
        self.pool_block = nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.AvgPool2d(kernel_size=8, stride=8, padding=0))
        
        # Return mu and logvar for reparameterization trick
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)
        
    def forward(self, x):
        # (N, 3, 128, 128) -> (N, 64, 64, 64)
        out = self.conv(x)
        # (N, 64, 64, 64) -> (N, 128, 32, 32) -> (N, 192, 16, 16) -> (N, 256, 8, 8)
        out = self.res_blocks(out)
        # (N, 256, 8, 8) -> (N, 256, 1, 1)
        out = self.pool_block(out)
        # (N, 256, 1, 1) -> (N, 256)
        out = out.view(x.size(0), -1)
        
        # (N, 256) -> (N, z_dim) x 2
        mu = self.fc_mu(out)
        log_var = self.fc_logvar(out)
        
        return (mu, log_var)