#!/usr/bin/env python
# coding: utf-8

# # Vanilla U-Net
# ----
# 
# As described by [Ronneberger's MICCAI 2015 paper](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28).

# In[1]:


import torch
from torch import nn
# from torchsummary import summary
import torch.nn.functional as F

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device:{device}")


# ## A double convolution block
# 
# Each stage in the U-Net, there are twice convolutions:

# In[2]:


class DoubleConvBlock(nn.Module):
    """
    ==> conv ==> BN ==> relu ==> conv ==> BN ==> relu ==>
    in_channels --> out_channels --> out_channels
    
    """
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_image = None
        
    def forward(self, in_image):
        self.out_image = self.model(in_image)
        return self.out_image
    
    def get_output(self):
        return self.out_image


# In[3]:


# summary(DoubleConvBlock(64,128).to(device), (64, 284, 284))


# ## Down sampling
# 
# At the encoding side (down sampling), it uses max pooling layer and then pass the output to the double convolution block.

# In[4]:


class DownSampleBlock(nn.Module):
    """
    (in_channel) ==> MaxPool
                        |
                        ==> DoubleConvBlock ==> (out_channel)
    """
    
    def __init__(self, in_channel, out_channel):
        super(DownSampleBlock, self).__init__()
        
        self.model = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvBlock(in_channel, out_channel)
        )
        
    def forward(self, in_image):
        return self.model(in_image)
    
    def get_output(self):
        return self.model[1].get_output()


# In[5]:


# summary(DownSampleBlock(64, 128).to(device), (64, 568, 568))


# ## Up sampling
# 
# At the decoding side, there is up sampling layer (ConvTranspose2d or Upsample), concat it with the output from the encoding side using a skip connection, then pass it to another DoubleConvBlock. Hence, the forward function contains 2 input arguments: the input image and the skip image.

# In[6]:


class UpSampleBlock(nn.Module):
    """
    up-sampling ==> concat ==> double-convolution
    """
    
    def __init__(self, in_channel):
        super(UpSampleBlock, self).__init__()
        
        self.skip_image = None
        self.up_sample = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=2, stride=2)
        self.double_conv = DoubleConvBlock(in_channel, in_channel//2)
        
    def forward(self, in_image, skip_image):
        
        # up sample the input image
        out_image = self.up_sample(in_image)
        
        # skip image has additional pad size due to ConvTranspose2d output
        pad_size = [
            item for sublist in 
            torch.tensor(out_image.shape[-2:]) - torch.tensor(skip_image.shape[-2:]) 
            for item in [sublist.item() // 2] * 2]
        
        # the concatenation
        out_image = torch.cat((F.pad(skip_image, pad_size), out_image), dim=1)
        
        # pass through the convolution
        out_image = self.double_conv(out_image)
        
        return out_image
    


# In[7]:


# summary(UpSampleBlock(1024).to(device), [(1024, 28, 28), (512, 64, 64)])


# ## Stiching up
# 
# Finally, we stich them all together creating the final U-Net architecture.

# In[8]:


class VanillaUNet(nn.Module):
    """
    Encoding: DoubleConvEncoder + 4 * DownSample
    Decoding:
    """
    
    def __init__(self, in_channel=1, out_channel=2):
        """
        Strict from Rossenberger's UNet: input = 1 (greyscale), output = 2 (mask)
        """
        super(VanillaUNet, self).__init__()
        
        self.layer_names = ['layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5']
        
        self.encoders = nn.ModuleDict([
            (name, block(ic, oc)) for name, block, ic, oc in zip(
                self.layer_names, 
                [DoubleConvBlock, DownSampleBlock, DownSampleBlock, 
                 DownSampleBlock, DownSampleBlock],
                [in_channel, 64, 128, 256, 512],
                [64, 128, 256, 512, 1024]
            )
        ])
        
        self.decoders = nn.ModuleDict([
            (name, UpSampleBlock(ic)) for name, ic in zip(
                self.layer_names,
                [128, 256, 512, 1024]
            )
        ])
        
        self.output_layer = nn.Conv2d(64, 2, kernel_size=1)
        
    def forward(self, in_image):
        out_image = in_image
        
        # encoding
        for k in self.layer_names:
            out_image = self.encoders[k](out_image)
            
        # decoding (from the back)
        for k in list(self.decoders.keys())[len(self.decoders)::-1]:
            out_image = self.decoders[k](out_image, self.encoders[k].get_output())
            
        # final layer
        out_image = self.output_layer(out_image)
        
        return out_image


# In[9]:


# summary(VanillaUNet().to(device), (1, 572, 572))


# In[ ]:




