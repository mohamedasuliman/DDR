#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:18:30 2022

@author: Mohamed A. Suliman

email: mohamedabdall78@hotmail.com
"""

import torch
import nibabel as nb
import numpy as np

## DDR dataloader function ##

class MRIImages(torch.utils.data.Dataset):

    def __init__(self, moving_dir, target_dir, moving_suffix, target_prefix, target_suffix,
                 Id_file1=None, Id_file2=None, Id_file3=None):

        self.subjects = []
        if Id_file1 is not None:
            self.subjects = self.subjects + open(Id_file1, "r").read().splitlines()
        if Id_file2 is not None:
            self.subjects = self.subjects + open(Id_file2, "r").read().splitlines()
        if Id_file3 is not None:
            self.subjects = self.subjects + open(Id_file3, "r").read().splitlines()
            
        self.moving_images = [] 
        self.target_images = [] 
        
        self.target_img=(nb.load(target_dir+target_prefix+target_suffix).darrays[0].data).astype(np.float32)

        for subject_id in self.subjects:
            moving_image = nb.load(moving_dir+str(subject_id)+moving_suffix).darrays[0].data
            self.moving_images.append(moving_image) 
            
    def __getitem__(self, index):
        
        moving_img = (self.moving_images[index]).astype(np.float32)
        moving_img = torch.transpose(torch.Tensor(moving_img).reshape(1,-1),0,1)
        target_img = torch.transpose(torch.Tensor(self.target_img).reshape(1,-1),0,1)
        return moving_img, target_img

    def __len__(self):
        return len(self.subjects)