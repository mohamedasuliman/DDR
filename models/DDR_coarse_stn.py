#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:26:04 2021

@author: Mohamed A. Suliman

email: mohamedabdall78@hotmail.com
"""

'''
DDR spatial transformer + CRF networks
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nb
from utils.utils import lat_lon_img, get_ico_center 
from utils.utils import icosphere_upsampling, bilinear_sphere_resample

### Set parameters ###

ver_dic = {0:12,1:42,2:162,3:642,4:2562,5:10242,6:40962,7:163842}
eps = 1e-12       
ico_radius = 100.0

ico_dir = 'icosphere/r100/'  # icoshpere directory. The icosphere named as: ico-level.surf.gii, level= 1 2,...etc
ddr_files_dir = 'DDR_files/' # DDR files directory
#################################################################

def load_ddr_var(data_ico):
    global upsamples
        
    upsamples=[]
    for i in range(data_ico):
        upsamples.append(torch.LongTensor(np.load(ddr_files_dir+'upsample_to_ico'+str(data_ico-i)+'.npy')))


class STN(nn.Module):
    def __init__(self, data_ico, labels_ico, control_ico, num_labels, device):
        '''
        Spatial Transformer Network
        
        Parameters
        ----------
        data_ico : int
            input data ico level (e.g., 6).
        labels_ico : int
            label grid ico level (e.g., 5).
        control_ico : int
            control grid ico level (e.g., 2). Note control_ico <= labels_ico
        num_labels : int
            number of labels around each control point (at label grid resolution).
        device : str
            CUDA or cpu.

        Returns
        -------
        warped_moving_img: moving image features registered to the target image -> tensor: ver_dic[data_ico] * int(in_ch/2)
        
        def_cont_ico: deformed control grid -> tensor: ver_dic[control_ico] * 3
        
        data_ico_warps: warps applied on the moving image grid -> tensor: ver_dic[data_ico] * 3        
        '''
        super(STN, self).__init__()
        
        self.device = device
        
        self.data_ico  = data_ico
        self.data_ver  = ver_dic[data_ico]
        
        self.labels_ico  = labels_ico
        
        self.control_ico = control_ico
        self.control_ver = ver_dic[control_ico]
        
        self.num_labels = num_labels
                        
        indices_matrix = torch.load('neigh_ver/indices_'+str(control_ico)+'_'+str(labels_ico))
        
        indices_neigh = indices_matrix[:,0:self.num_labels]
        
        ###  load icos  ###
        self.icosphere_labels_temp =  torch.Tensor(nb.load(ico_dir+'ico-'+
                                                           str(labels_ico)+'.surf.gii').darrays[0].data).to(self.device)                 
        self.icosphere_labels = self.icosphere_labels_temp[indices_neigh,:]
               
        self.icosphere_control =  torch.Tensor(nb.load(ico_dir+'ico-'+
                                                       str(control_ico)+'.surf.gii').darrays[0].data).to(self.device)       
        self.icosphere_data =  torch.Tensor(nb.load(ico_dir+'ico-'+
                                                    str(data_ico)+'.surf.gii').darrays[0].data).to(self.device)         
        load_ddr_var(self.data_ico)
        
        self.model_CRF = CrfRnn(self.num_labels, self.control_ver, self.device)
                
          
    def forward(self, moving_img, def_idxs):

        def_idxs_norm = F.normalize(def_idxs.clone(), p=4, dim=1)
                
        temp_def_cont_ico = torch.bmm(def_idxs_norm.unsqueeze(1), self.icosphere_labels).squeeze(1)
                        
        temp_def_cont_ico = temp_def_cont_ico/(torch.norm(temp_def_cont_ico,
                                                                dim=1, keepdim=True).repeat(1,3)+eps)        

        ## apply CRF network ##

        new_def_idxs= self.model_CRF(def_idxs_norm, temp_def_cont_ico)#smooth_filter) 
        
        def_cont_ico = torch.bmm(new_def_idxs.unsqueeze(1), self.icosphere_labels).squeeze(1)
       
        def_cont_ico = def_cont_ico/(torch.norm(def_cont_ico, dim=1, keepdim=True).repeat(1,3)+eps)
             

        current_def_ico = def_cont_ico.clone()
        num_ver = self.control_ver
        
        ico_idx = 1
        while num_ver < self.data_ver:
            current_def_ico = icosphere_upsampling(num_ver, current_def_ico, num_ver*4-6, 
                                                   upsamples[self.data_ico-(self.control_ico+ico_idx)],
                                                   device=self.device)
            num_ver = num_ver*4-6
            ico_idx+= 1
                   
        temp_def_data_ico  = current_def_ico/(torch.norm(current_def_ico, dim=1, keepdim=True).repeat(1,3)+eps)
        
        ico_center = get_ico_center(temp_def_data_ico,device=self.device)
    
        def_data_ico = ico_radius*(temp_def_data_ico-ico_center) 
        
        data_ico_warps = def_data_ico-self.icosphere_data
        
        img = lat_lon_img(moving_img,device=self.device)

        warped_moving_img = bilinear_sphere_resample(def_data_ico, img, radius=ico_radius,
                                                     device=self.device) 
                
        return warped_moving_img, def_cont_ico, data_ico_warps
   

class CrfRnn(nn.Module):
    def __init__(self, num_labels, control_ver, device, num_iter=5):        
        '''
        Parameters
        ----------
        num_labels : int
            number of labels around each control point (at label grid resolution).
        control_ver : int
            number of control points.        
        device : str
            CUDA or cpu
        num_iter : int, optional
            number of CRF RNN iterations. The default is 5.

        Returns
        -------
        def_idxs_new: updated deformation indices. (tensor: control_ver * num_labels)
        '''       
        super(CrfRnn, self).__init__()
         
        self.control_ver = control_ver
        
        self.device = device
        self.num_iter = num_iter        
        self.gamma = 0.5
        
        self.filter_weights = nn.Parameter(1.0*torch.ones(control_ver, control_ver))
                
        self.spatial_weights = nn.Parameter(4.0*torch.eye(control_ver, dtype=torch.float32))

        self.comp_matrix = nn.Parameter(torch.eye(num_labels, dtype=torch.float32))

        self.softmax_layer = nn.Softmax(dim=1)
        
    def forward(self, def_idxs, def_cont_ico_crf):
        
        smooth_filter = torch.zeros((self.control_ver,self.control_ver), device=self.device)
                
        for i in range(self.control_ver):
            
            alpha = (torch.norm((def_cont_ico_crf[i,:]-def_cont_ico_crf),dim=1)**2).reshape(1,-1)
          
            smooth_filter[i,:] = torch.exp(-alpha/(2.0*self.gamma**2))           
   
        filter_mat = smooth_filter.fill_diagonal_(0, wrap=False)
               
        U = torch.log(torch.clamp(def_idxs, 1e-6, 1))
        
        def_idxs_new = self.softmax_layer(U)
        
        filter_mat = self.filter_weights*filter_mat
        
        for iter in range(self.num_iter):
            
            phi_t = torch.mm(filter_mat,def_idxs_new) 
            
            phi = - torch.mm(self.spatial_weights,phi_t)
            
            phi = torch.mm(phi,self.comp_matrix)
            
            phi = U - phi
            
            def_idxs_new = self.softmax_layer(phi)
            

        return def_idxs_new
