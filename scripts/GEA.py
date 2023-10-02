from typing import Any
from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

class GaussianErasing(object):
    
    def __init__(self,probability=0.5,sl=0.09,sh=0.7,ratiol=1,ratioh=3,thetal=-89,thetah=89,mean=[0.4914, 0.4822, 0.4465],density=0.5,scalar=10) -> None:
        '''     
        Param:
            probability: Probability of triggering GEA
            sl: Lower limit of occlusion area
            sh: Upper limit of the occlusion area
            ratiol: Lower limit of sigma2/sigma1 
            ratiol: Upper limit of sigma2/sigma1 
            thetal: Lower limit of rotation angle
            thetah: Upper limit of ratation angle
            mean: [] value of replace pixel, default = [0.4914, 0.4822, 0.4465] which is the mean of ImageNet dataset
            
            density: Density of Gaussian sampling
            scalar: Mapping scale of sampling points and image coordinates 
        '''
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.ratiol = ratiol
        self.ratioh = ratioh
        self.thetal = thetal
        self.thetah = thetah
        self.mean = mean
        self.density = density
        self.scalar = scalar
    
    def __call__(self, img) -> Any:
        if random.uniform(0,1) > self.probability:
            return img

        
        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            
            # Randomly generate parameters
            target_area = random.uniform(self.sl,self.sh) * area
            aspect_ratio = random.uniform(self.ratiol,self.ratioh)
            theta = random.uniform(self.thetal,self.thetah) * np.pi / 180   
    
            w = np.sqrt(target_area / aspect_ratio) / 5         
            h = np.sqrt(target_area * aspect_ratio) / 5         
            
            # sigma1 is fixed as 1.0, calculate sigma2 by sigma1 * ratio
            sigma1 = 1
            sigma2 = aspect_ratio * sigma1
            
            
            # calculate the legal occlusion area scale
            phi = np.arctan(1 / aspect_ratio)
            hypotenuse = np.sqrt((5*w)**2+(5*h)**2)   
            if theta >= 0:
                W = hypotenuse * np.sin(theta+phi)
                H = hypotenuse * np.cos(theta-phi)
            else :
                _theta = -1 * theta
                W = hypotenuse * np.sin(_theta+phi)
                H = hypotenuse * np.cos(_theta-phi)
            W = int(W)
            H = int(H)
            
            # generate mask
            mask = generate_gaussian_mask(sigma1=sigma1,sigma2=sigma2,theta=theta,w=W,h=H,density=self.density,scalar=self.scalar)
                                 
                                   
            if W // 2 < img.size()[2] and H // 2 < img.size()[1]:
                # generate expand image
                expand_img = torch.zeros(img.size()[0],img.size()[1]+H,img.size()[2]+W)
                expand_img[:,H//2:H//2+img.size()[1],W//2:W//2+img.size()[2]] = img[:,:,:]
                
                x1 = random.randint(0, expand_img.size()[1] - H)
                y1 = random.randint(0, expand_img.size()[2] - W)
                
                # erase pixel by mask
                if expand_img.size()[0] == 3:
                    
                    expand_img[:, x1:x1+H, y1:y1+W] = expand_img[:, x1:x1+H, y1:y1+W] * (1-mask)
                    add_mask = (torch.ones(mask.size())).repeat(3,1,1)
                    add_mask = add_mask * mask
                    add_mask[0,:,:] = add_mask[0,:,:] * self.mean[0]
                    add_mask[1,:,:] = add_mask[1,:,:] * self.mean[1]
                    add_mask[2,:,:] = add_mask[2,:,:] * self.mean[2]
                    expand_img[:, x1:x1+H, y1:y1+W] = expand_img[:, x1:x1+H, y1:y1+W] + add_mask
                else:
                    expand_img[0, x1:x1+H, y1:y1+W] = expand_img[0, x1:x1+H, y1:y1+W] * (1-mask)
                    add_mask = (torch.ones(mask.size())).repeat(1,1,1)
                    add_mask = add_mask * mask
                    add_mask[0,:,:] = add_mask[0,:,:] * self.mean[0]
                    expand_img[:, x1:x1+H, y1:y1+W] = expand_img[:, x1:x1+H, y1:y1+W] + add_mask
                
                img[:,:,:] = expand_img[:,H//2:H//2+img.size()[1],W//2:W//2+img.size()[2]]
                
                return img
        
        return img

def generate_gaussian_mask(sigma1,sigma2,theta,w,h,density=0.5,scalar=10):
    
    w = int(w)
    h = int(h)
    
    if np.abs(theta) > 3.14:
        theta = theta * np.pi / 180
    
    # scalar matrix
    scalarMatrix=np.dot(np.matrix([[sigma1**2,0],[0,sigma2**2]]),np.identity(2))

    # rotation matrix
    rotationMatrix=np.matrix([[np.cos(theta),-1*np.sin(theta)],
                            [np.sin(theta),np.cos(theta)]])
    
    # covariance matrix
    covMatrix=np.dot(np.dot(rotationMatrix,scalarMatrix),rotationMatrix.transpose()) 
    
    pts = np.random.multivariate_normal([0, 0], covMatrix, size=int(w*h*density))
    X = torch.Tensor(pts[:,0])      
    Y = torch.Tensor(pts[:,1])      
    locs = torch.stack((Y,X),dim=1)
    
    # mapping
    pts = (locs * scalar).int()            
    pts[:,0] = pts[:,0] + h //2     # h
    pts[:,1] = pts[:,1] + w //2     # w
        
    select_mask = (pts[:,0]>=0)&(pts[:,0]<h)&(pts[:,1]>=0)&(pts[:,1]<w)
    pts = pts[select_mask]              
    pts = torch.unique(pts,dim=0)       
    
    mask = torch.zeros(h,w)   
    lx = torch.LongTensor(pts[:,0].numpy()) 
    ly = torch.LongTensor(pts[:,1].numpy()) 
    replace_value = torch.ones_like(pts[:,0],dtype=mask.dtype)
    mask = mask.index_put((lx,ly),replace_value)
    
    return mask
    

        
if __name__ == '__main__':
    
    img = Image.open('../img/test.png')

    img.show()
    
    trans = transforms.Compose([
                transforms.ToTensor(),
                GaussianErasing(probability=1.0,thetal=0)
                ]) 
    
    img1 = trans(img)
    img1 = transforms.ToPILImage()(img1)
    img1.show()

        