#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:59:54 2022

@author: Mohamed A. Suliman

email: mohamedabdall78@hotmail.com
"""

import torch
import torch.nn as nn
import nibabel as nb
from utils.utils import lat_lon_img, bilinear_sphere_resample

ico_dir = 'icosphere/r100/'  # icoshpere directory. The icosphere named as: ico-level.surf.gii, level= 1 2,...etc


   
class STN(nn.Module):
    def __init__(self, data_ico, device, interpolation_mode='bilinear'):
        '''        
        Parameters
        ----------
        device : str
            CUDA or cpu.
        interpolation_mode : str, optional
            The default is 'bilinear'.

        Returns
        -------
        rotated_img : resampled moving image on a rotated icosphere.

        '''
        super().__init__()
        
        self.inter_mode = interpolation_mode
        self.device = device
        
        icosphere =  nb.load(ico_dir+'ico-'+str(data_ico)+'.surf.gii') # Load your icosphere

        icosphere_coords = icosphere.darrays[0].data
        
        grid  = torch.Tensor(icosphere_coords)         
        w_dim = torch.ones((grid.shape[0], 1))
        grid  = torch.cat((grid, w_dim), 1)
        
        self.grid_T = torch.transpose(grid, 0, 1)

                
    def forward(self, moving_img, rot_matrix):
                                 
        rot_grid = torch.matmul(rot_matrix,self.grid_T)
        rot_grid = torch.transpose(rot_grid,0,1) 
                      
        img = lat_lon_img(moving_img, self.device)
        
        radius = torch.max(rot_grid[:,0]) 
        
        if self.inter_mode =='bilinear':
            rotated_img = bilinear_sphere_resample(rot_grid[:,0:3], img, radius=radius, device=self.device)      
        else:
            print('Error: Unsupported interpolation mode')

        return rotated_img  