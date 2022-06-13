#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:44:18 2022

@author: Mohamed A. Suliman

email: mohamedabdall78@hotmail.com
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch_geometric.nn as gnn
from .DDR_affine_stn import STN


ver_dic = {0:12,1:42,2:162,3:642,4:2562,5:10242,6:40962,7:163842}

ddr_files_dir = 'DDR_files/'  # DDR files directory


def gmm_conv(in_ch, out_ch, kernel_size=3):
    return gnn.GMMConv(in_ch, out_ch, dim=2, kernel_size=kernel_size)
    
   
class DDR_affine(nn.Module):
    def __init__(self, in_ch, out_ch, num_features, data_ico, device, conv_layer=gmm_conv,
                 activation_layer=nn.LeakyReLU(0.2, inplace=True)):
        super(DDR_affine, self).__init__()
        
        self.conv_layer = conv_layer
        self.device = device
        self.data_ico = data_ico
        
        num_ver = ver_dic[self.data_ico]
        
        self.in_channels  = in_ch
        self.out_channels = out_ch
        
        self.pseudo_in = torch.Tensor(np.load(ddr_files_dir+'pseudo_'+str(self.data_ico)+'.npy'))
                
        self.conv1 = conv_layer(self.in_channels, num_features[0])        
        self.conv2 = conv_layer(num_features[0], num_features[1])       
        self.conv3 = conv_layer(num_features[1], num_features[2])   
        self.conv4 = conv_layer(num_features[2], num_features[3])
        # self.conv4 = conv_layer(self.in_channels+num_features[2], num_features[3])

        self.activation_layer = activation_layer
                       
        self.out = nn.Sequential(
            nn.Linear(num_features[3]*num_ver, self.out_channels),
            activation_layer)
        
        self.transformer = STN(self.data_ico, device=self.device)
        
        self.rot_layer = nn.Linear(self.out_channels,6)
        
        
        # initialize rotation layer 
        
        self.rot_matrix = torch.tensor([1, 0, 0, 0, 0, 1, 0 ,0, 0,0,1,0,0,0,0,1], dtype=torch.float).view(4, 4)
        
        self.rot_layer.weight.data.zero_()
        
        self.rot_layer.bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))


    def forward(self, moving_img, target_img, edge_input):
        
        x_in =  torch.cat([moving_img, target_img], dim=1) 
        
        x = self.conv1(x_in,edge_input, self.pseudo_in.to(self.device)) 
        x = self.activation_layer(x)
                
        x = self.conv2(x,edge_input, self.pseudo_in.to(self.device)) 
        x = self.activation_layer(x)
                
        x = self.conv3(x,edge_input, self.pseudo_in.to(self.device)) 
        x = self.activation_layer(x)
        
        # x = torch.cat([x,x_in], dim = 1)
        x = self.conv4(x,edge_input, self.pseudo_in.to(self.device)) 
        x = self.activation_layer(x)
               
        
        #### Rotation Stage ####
        x = x.reshape(1,-1)
        x = self.out(x)

        self.rot_matrix  = self.get_rotation_matrix(x)
        rotated_img = self.transformer(moving_img, self.rot_matrix)
                            
        return rotated_img, self.rot_matrix 

        
    def get_rotation_matrix(self, u):
        v = self.rot_layer(u)
                            
        x = v[:,0:3]
        y = v[:,3:6]
                
        x = normalize(x, p=2.0, dim = 1) 
        z = torch.cross(x,y) 
        z = normalize(z, p=2.0, dim = 1) 
        y = torch.cross(z,x)
                 
        x = x.view(-1,3,1)        
        y = y.view(-1,3,1)
        z = z.view(-1,3,1)
        
        rot_matrix_3x3 = torch.cat((x,y,z), 2) 
        rot_matrix_4x4 = get_4x4_rotation_matrix(rot_matrix_3x3)         
                
        return rot_matrix_4x4
 
        
def get_4x4_rotation_matrix(rot_matrix_3x3):
    batch_size = rot_matrix_3x3.shape[0]
    
    row4 = torch.autograd.Variable(torch.zeros(batch_size,1,3))
    col4 = torch.autograd.Variable(torch.zeros(batch_size,4,1))
    
    col4[:,3,0]=col4[:,3,0]+1
    
    rot_matrix_4x3 = torch.cat((rot_matrix_3x3, row4),1)
    rot_matrix_4x4 = torch.cat((rot_matrix_4x3, col4),2)    
    return rot_matrix_4x4.squeeze(0)
