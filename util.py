import torch
from torch.autograd import Variable

'''
    < var >
    Convert tensor to Variable
'''
def var(tensor, requires_grad=True):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    var = Variable(tensor.type(dtype), requires_grad=requires_grad)
    
    return var

'''
    < make_img >
    Generate images

    * Parameters
    dloader : Data loader for test data set
    G : Generator
    z : random_z(size = (N, img_num, z_dim))
        N : test img number / img_num : Number of images that you want to generate with one test img / z_dim : 8
    img_num : Number of images that you want to generate with one test img
'''
def make_img(dloader, G, z, img_num=5, img_size=128):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    dloader = iter(dloader)
    img, _ = dloader.next()

    N = img.size(0)    
    img = var(img.type(dtype))

    result_img = torch.FloatTensor(N * (img_num + 1), 3, img_size, img_size).type(dtype)

    for i in range(N):
        # original image to the leftmost
        result_img[i * (img_num + 1)] = img[i].data

        # Insert generated images to the next of the original image
        for j in range(img_num):
            img_ = img[i].unsqueeze(dim=0)
            z_ = z[i, j, :].unsqueeze(dim=0)
            
            out_img = G(img_, z_)
            result_img[i * (img_num + 1) + j + 1] = out_img.data


    # [-1, 1] -> [0, 1]
    result_img = result_img / 2 + 0.5
    
    return result_img