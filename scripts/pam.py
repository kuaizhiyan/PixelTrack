 
import torch
import torch.nn as nn
import numpy as np
 
class PAM(nn.Module):

    def __init__(self,in_planes,kernel_size=3,ratio=16,groups=1) -> None:
        super().__init__()

        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels=in_planes,out_channels=in_planes,kernel_size=kernel_size,padding=1,groups=self.groups,stride=1)

        self.fc1 = nn.Conv2d(in_channels=in_planes,out_channels=in_planes // ratio,kernel_size=3,padding=1)
        self.fc2 = nn.Conv2d(in_channels=in_planes//ratio,out_channels=in_planes,kernel_size=3,padding=1)
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.gn =torch.nn.GroupNorm(num_groups=groups,num_channels=in_planes)
    
    def channel_shuffle(self, x):
      
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.groups == 0 
        group_channels = num_channels // self.groups
        
        x = x.reshape(batchsize, group_channels, self.groups, height, width)
        x = x.permute(0, 2, 1, 3, 4) 
        x = x.reshape(batchsize, num_channels, height, width)
        
        return x
    
    def forward(self,x):

      # channel shuffle
      s_x = self.channel_shuffle(x)

      # part-wise learning
      spatial_attention = (self.gn(self.conv1(s_x)))

      # pixel-wise learning
      spatial_attention = self.fc1(spatial_attention)
      spatial_attention = self.fc2(spatial_attention)
      spatial_attention = self.sigmoid(spatial_attention)

      # element-wise product
      out = s_x * spatial_attention

      return out