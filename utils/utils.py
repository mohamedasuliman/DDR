#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  10 18:07:54 2022

@author: Mohamed A. Suliman

email: mohamedabdall78@hotmail.com
"""
import numpy as np
import torch
import math
import nibabel as nb
from nibabel.gifti import GiftiImage, GiftiDataArray, GiftiCoordSystem, GiftiMetaData,  GiftiNVPairs
from nibabel.filebasedimages import FileBasedHeader

ico_dir = 'icosphere/r100/'  
ddr_files_dir = 'DDR_files/' 

ver_ico_dic = {12:0,42:1,162:2,642:3,2562:4,10242:5,40962:6,163842:7}


def LossFuns(x, y):
    
    loss_mse = torch.mean(torch.pow((x - y), 2))
    loss_cc  = 1 - ((x - x.mean()) * (y - y.mean())).mean() / x.std() / y.std()
    
    return loss_mse, loss_cc 


def grad_loss(warps, hex_org, device, reg_fun='l1'):
    weights = torch.ones((7, 1), dtype=torch.float32, device = device)
    weights[6] = -6 

    warps_dx = torch.abs(torch.mm(warps[:,[0]][hex_org].view(-1, 7), weights))    
    warps_dy = torch.abs(torch.mm(warps[:,[1]][hex_org].view(-1, 7), weights))    
    warps_dz = torch.abs(torch.mm(warps[:,[2]][hex_org].view(-1, 7), weights))

    if reg_fun == 'l2':
        warps_dx = warps_dx*warps_dx
        warps_dy = warps_dy*warps_dy
        warps_dz = warps_dz*warps_dz

    warps_gradient = (torch.mean(warps_dx) + torch.mean(warps_dy) + torch.mean(warps_dz))/3.0

    return warps_gradient  


def save_gifti(icosphere_array, subject_id, file_to_save): 
       
    ico_level= ver_ico_dic[icosphere_array.shape[0]]
    
    file_name= file_to_save+str(subject_id)+'.DDR.ico-'+str(ico_level)+'.surf.gii'
        
    ico_faces = nb.load(ico_dir+'ico-'+str(ico_level)+'.surf.gii').darrays[1].data 
        
    data_cor = GiftiCoordSystem(dataspace='NIFTI_XFORM_TALAIRACH',
                                xformspace='NIFTI_XFORM_TALAIRACH',
                                xform=None)
 
    nvpair1 = GiftiNVPairs(name='AnatomicalStructureSecondary', value='Invalid')
    nvpair2 = GiftiNVPairs(name='GeometricType', value='Spherical')
     
    Gifti_meta_data=  GiftiMetaData(nvpair1)
    Gifti_meta_data.data.insert(1, nvpair2)
    
    Gifti_data = GiftiDataArray(data= icosphere_array,
                                intent='NIFTI_INTENT_POINTSET',
                                datatype='NIFTI_TYPE_FLOAT32', 
                                encoding='GIFTI_ENCODING_B64GZ', 
                                endian='little', 
                                coordsys=data_cor, 
                                ordering='RowMajorOrder', 
                                meta=Gifti_meta_data, 
                                ext_fname='', 
                                ext_offset=0)
    
    Gifti_cor_face = GiftiCoordSystem(dataspace='NIFTI_XFORM_UNKNOWN', 
                                      xformspace='NIFTI_XFORM_UNKNOWN', 
                                      xform=None)
    
    Gifti_meta_face =  GiftiMetaData()
    
    Gifti_face = GiftiDataArray(data=ico_faces, 
                                intent='NIFTI_INTENT_TRIANGLE', 
                                datatype='NIFTI_TYPE_INT32', 
                                encoding='GIFTI_ENCODING_B64GZ', 
                                endian='little', 
                                coordsys=Gifti_cor_face, 
                                ordering='RowMajorOrder', 
                                meta=Gifti_meta_face, 
                                ext_fname='', 
                                ext_offset=0)
    
    file_head = FileBasedHeader()
    
    my_ico = GiftiImage(header=file_head,
                        extra=None,
                        file_map=None,
                        meta=None,
                        labeltable=None,
                        darrays=[Gifti_data]+[Gifti_face],
                        version='1.0')
    
    nb.save(my_ico, file_name)
    print("Subject '{}' saved!".format(subject_id))
    

def icosphere_upsampling(num_ver, current_ico, next_ver, hex_i, device):
 
    assert current_ico.shape[1] == 3, "icosphere.shape[1] must equal 3"
    assert next_ver == num_ver*4-6, "next_ver ≠ num_ver*4-6"
    
    next_ico = torch.zeros((next_ver, 3), dtype=torch.float, device=device)
    
    next_ico[:num_ver] = current_ico
    next_ico[num_ver:] = torch.mean(current_ico[hex_i],dim=1)

    return next_ico

def get_ico_center(ico_ver,device):
    r_min = torch.min(ico_ver,dim=0)[0].to(device)
    r_max = torch.max(ico_ver,dim=0)[0].to(device)    
    ico_center = (r_min+r_max)/2.0    
    return ico_center


'''
The lat_lon_img Func and the bilinear_sphere_resample Func 
are both inspired by the source code in:   
https://github.com/zhaofenqiang/Spherical_U-Net
'''    

def lat_lon_img(moving_feat, device):
    
    num_ver = len(moving_feat)
    
    img_idxs = np.load(ddr_files_dir+'img_indices_'+ str(num_ver) +'.npy').astype(np.int64)
    img_weights = np.load(ddr_files_dir+'img_weights_'+ str(num_ver) +'.npy').astype(np.float32)
    
    img_idxs =torch.from_numpy(img_idxs).to(device)
    img_weights = torch.from_numpy(img_weights).to(device)    

    W = int(np.sqrt(len(img_idxs)))
    
    img = torch.sum(((moving_feat[img_idxs.flatten()]).reshape(img_idxs.shape[0], img_idxs.shape[1], moving_feat.shape[1]))*((img_weights.unsqueeze(2)).repeat(1,1,moving_feat.shape[1])),1)
    
    img = img.reshape(W, W, moving_feat.shape[1])
    
    return img
            

def bilinear_sphere_resample(rot_grid, org_img, radius, device):
        
    assert rot_grid.shape[1] == 3, "grid.shape[1] ≠ 3"
    
    rot_grid_r1 = rot_grid/radius
    
    w = org_img.shape[0]

    rot_grid_r1[:,2] = torch.clamp(rot_grid_r1[:,2].clone(), -0.9999999, 0.9999999)
    
    Theta = torch.acos(rot_grid_r1[:,2]/1.0)    
    Phi = torch.zeros_like(Theta)
    
    zero_idxs = (rot_grid_r1[:,0] == 0).nonzero(as_tuple=True)[0]
    rot_grid_r1[zero_idxs, 0] = 1e-15
    
    pos_idxs = (rot_grid_r1[:,0] > 0).nonzero(as_tuple=True)[0]
    Phi[pos_idxs] = torch.atan(rot_grid_r1[pos_idxs, 1]/rot_grid_r1[pos_idxs, 0])
    
    neg_idxs = (rot_grid_r1[:,0] < 0).nonzero(as_tuple=True)[0]
    Phi[neg_idxs] = torch.atan(rot_grid_r1[neg_idxs, 1]/rot_grid_r1[neg_idxs, 0]) + math.pi
     
    Phi = torch.remainder(Phi + 2 * math.pi, 2*math.pi)
    
    assert len(pos_idxs) + len(neg_idxs) == len(rot_grid_r1)
    
    u = Phi/(2*math.pi/(w-1))
    v = Theta/(math.pi/(w-1))
        
    v = torch.clamp(v, 0.0000001, org_img.shape[1]-1.00000001).to(device)
    u = torch.clamp(u, 0.0000001, org_img.shape[1]-1.1).to(device)
    
    u_floor = torch.floor(u)
    u_ceil = u_floor + 1
    v_floor = torch.floor(v)
    v_ceil = v_floor + 1
    
    img1 = org_img[v_floor.long(), u_floor.long()]
    img2 = org_img[v_floor.long(), u_ceil.long()]
    img3 = org_img[v_ceil.long() , u_floor.long()]     
    img4 = org_img[v_ceil.long() , u_ceil.long()]
    
    Q1 = (u_ceil-u).unsqueeze(1)*img1 + (u-u_floor).unsqueeze(1)*img2    
    Q2 = (u_ceil-u).unsqueeze(1)*img3 + (u-u_floor).unsqueeze(1)*img4    
    Q  = (v_ceil-v).unsqueeze(1)*Q1 + (v-v_floor).unsqueeze(1)*Q2
       
    return Q 
    