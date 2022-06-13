#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:33:27 2021

@author: Mohamed A. Suliman

email: mohamedabdall78@hotmail.com
"""

'''
DDR coarse network
'''


import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from models.DDR_coarse_stn import STN

ver_dic  = {0:12,1:42,2:162,3:642,4:2562,5:10242,6:40962,7:163842}
ddr_files_dir = 'DDR_files/'  # DDR files directory

###################################################################

def gmm_conv(in_ch, out_ch, kernel_size=3):
    return gnn.GMMConv(in_ch, out_ch, dim=2, kernel_size=kernel_size)

def load_ddr_var(data_ico):
    global hexes
    global edge_indexes
    global pseudos
    global upsamples
    
    hexes=[]    
    for i in range(data_ico):
        hexes.append(torch.LongTensor(np.load(ddr_files_dir+'hexagons_'+str(data_ico-i)+'.npy')))
    
    edge_indexes=[]
    for i in range(data_ico):
        edge_indexes.append(torch.LongTensor(np.load(ddr_files_dir+'edge_index_'+str(data_ico-i)+'.npy')))
    
    pseudos=[]
    for i in range(data_ico):
        pseudos.append(torch.LongTensor(np.load(ddr_files_dir+'pseudo_'+str(data_ico-i)+'.npy')))
    
    upsamples=[]
    for i in range(data_ico):
        upsamples.append(torch.LongTensor(np.load(ddr_files_dir+'upsample_to_ico'+str(data_ico-i)+'.npy')))


class hex_upsample(nn.Module):
    def __init__(self, ico_level, data_ico, device):
        super(hex_upsample, self).__init__()
        
        self.hex = hexes[data_ico-ico_level].to(device)
        self.upsample = upsamples[data_ico-ico_level].to(device)
        self.device =  device       
        
    def forward(self, ico_feat):
        n_ver = int(ico_feat.shape[0])
        up_ico_feat = torch.zeros(self.hex.shape[0],ico_feat.shape[1]).to(self.device)
        up_ico_feat[:n_ver] = ico_feat
        up_ico_feat[n_ver:] = torch.mean(ico_feat[self.upsample],dim=1)
        
        return up_ico_feat

        
class hex_pooling(nn.Module):
    def __init__(self, ico_level, data_ico, device):
        super(hex_pooling, self).__init__()
        
        self.hex = hexes[data_ico-ico_level].to(device)
        
    def forward(self, ico_feat):
        n_ver = int((ico_feat.size()[0]+6)/4)
        num_feats = ico_feat.size()[1]

        ico_feat = ico_feat[self.hex[0:n_ver]].view(n_ver, num_feats, 7)
        ico_feat = torch.mean(ico_feat, 2)
                
        return ico_feat     

                       
class onering_conv_layer(nn.Module):
    """
    This is the Spherical U-Net conv filter
    Reproduced from: https://github.com/zhaofenqiang/Spherical_U-Net/
    Input: n_ver x in_feats 
    Output: n_ver x out_feats 
    """  
    def __init__(self, in_feats, out_feats, hex_in):
        super(onering_conv_layer, self).__init__()

        self.in_feats  = in_feats
        self.hex_in = hex_in
        self.weight = nn.Linear(7*in_feats, out_feats)
            
    def forward(self, x):
        x_new = (x[self.hex_in].view(len(x), 7*self.in_feats)).float()
        out_features = self.weight(x_new)        
        return out_features
 
       
class DDR_coarse(nn.Module):
    def __init__(self, in_ch, num_features, data_ico, labels_ico, control_ico, 
                 num_labels, device, conv_layer=gmm_conv,
                 activation_layer=nn.LeakyReLU(0.2, inplace=True)):
        '''
        Parameters
        ----------
        in_ch : int
            input channels.
        num_features : list
            learnable features at each conv layer.
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
        conv_layer : optional
            convolutional layer. The default is gmm_conv.
        activation_layer : torch.nn, optional
            non-linear activation layer. The default is nn.LeakyReLU(0.2, inplace=True).

        Returns
        -------
        warped_moving_img: moving image features registered to the target image -> tensor: ver_dic[data_ico] * int(in_ch/2)
        
        deformed_control_ico: deformed control grid -> tensor: ver_dic[control_ico] * 3
        
        warps_moving: warps applied on the moving image grid -> tensor: ver_dic[data_ico] * 3
        
        '''        
        super(DDR_coarse, self).__init__()
        
        self.conv_layer  = conv_layer       
        self.in_channels = in_ch
        self.data_ico = data_ico
        
        self.data_ver   = ver_dic[data_ico]       
        self.labels_ver = ver_dic[labels_ico]        
        self.control_ver= ver_dic[control_ico]
        
        self.device = device
        self.num_labels = num_labels
        self.level_dif = data_ico-control_ico
        
        self.activation_layer  = activation_layer        
        self.activation_layer_2 = nn.ReLU() 
        
        load_ddr_var(data_ico)
        
        #### Encoder layers ####
        
        self.conv1 = conv_layer(self.in_channels, num_features[0])
        self.conv1_s = conv_layer(num_features[0],num_features[0])
             
        self.conv2 = conv_layer(num_features[0], num_features[1])
        self.conv2_s = conv_layer(num_features[1],num_features[1])
                
        self.conv3 = conv_layer(num_features[1], num_features[2])
        self.conv3_s = conv_layer(num_features[2],num_features[2])
                
        self.conv4 = conv_layer(num_features[2], num_features[3])
        self.conv4_s = conv_layer(num_features[3],num_features[3])
              
        self.conv5 = conv_layer(num_features[3], num_features[4])
        self.conv5_s = conv_layer(num_features[4],num_features[4])        
        
        ##### Decoder layers ####
        
        self.conv11 = conv_layer(num_features[0] + self.in_channels, num_features[0])
        self.conv11_s = conv_layer(num_features[0],num_features[0])

        self.conv10 = conv_layer(num_features[1] + num_features[0], num_features[0])
        self.conv10_s = conv_layer(num_features[0],num_features[0])
        
        self.conv9 = conv_layer(num_features[2] + num_features[1], num_features[1])
        self.conv9_s = conv_layer(num_features[1],num_features[1]) 
        
        self.conv8 = conv_layer(num_features[3] + num_features[2], num_features[2])
        self.conv8_s = conv_layer(num_features[2],num_features[2])
        
        self.conv7 = conv_layer(num_features[4] + num_features[3], num_features[3])
        self.conv7_s = conv_layer(num_features[3],num_features[3]) 
        
        self.conv6 = conv_layer(num_features[4], num_features[4])
        self.conv6_s = conv_layer(num_features[4],num_features[4])
        
        #### #####
        self.down_to_control = nn.ModuleList([]) 
                
        for i in range(self.level_dif): 
            self.down_to_control.append(hex_pooling(data_ico-i, data_ico, device))
         
        ### pooling and upsampling layers ###   
           
        self.pool1 = hex_pooling(data_ico, data_ico, device)
        self.pool2 = hex_pooling(data_ico-1, data_ico, device)
        self.pool3 = hex_pooling(data_ico-2, data_ico, device)
        self.pool4 = hex_pooling(data_ico-3, data_ico, device)
        self.pool5 = hex_pooling(data_ico-4, data_ico, device)
                       
        self.upsample1 = hex_upsample(data_ico-5, data_ico, device)
        self.upsample2 = hex_upsample(data_ico-4, data_ico, device)
        self.upsample3 = hex_upsample(data_ico-3, data_ico, device)
        self.upsample4 = hex_upsample(data_ico-2, data_ico, device)
        self.upsample5 = hex_upsample(data_ico-1, data_ico, device)
        self.upsample6 = hex_upsample(data_ico, data_ico, device)
                 
        ### feedforward Spherical Network Layers ###
        
        label_feat = [num_features[0], num_labels]
        # label_feat = [num_features[0],256,self.num_labels]     
        
        self.conv_label = onering_conv_layer(label_feat[0],label_feat[1], hex_in=hexes[0])
        # self.conv_label2 = onering_conv_layer(label_feat[1],label_feat[2], hex_in=hexes[0])
         
        self.softmax_layer = nn.Softmax(dim=1)
         
        ### spatial transformer+CRF network ###
        
        self.transformer_crf = STN(data_ico=data_ico,
                                   labels_ico=labels_ico, 
                                   control_ico=control_ico, 
                                   num_labels=num_labels,
                                   device=device)        
        
        
    def forward(self, moving_img, target_img, edge_input):
        
        x_in =  torch.cat([moving_img, target_img], dim=1) 
        
        ########  ico-data_ico layer ########
        x = self.conv1(x_in,edge_input, pseudos[0].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv1_s(x,edge_input, pseudos[0].to(self.device)) 
        x = self.activation_layer(x)
        
        x1= self.pool1(x)
        
        ########  ico-(data_ico-1) layer ########
        x = self.conv2(x1,edge_indexes[1].to(self.device), pseudos[1].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv2_s(x,edge_indexes[1].to(self.device), pseudos[1].to(self.device))
        x = self.activation_layer(x)
               
        x2= self.pool2(x)
        
        ######## ico-(data_ico-2) layer ########       
        x = self.conv3(x2,edge_indexes[2].to(self.device), pseudos[2].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv3_s(x,edge_indexes[2].to(self.device), pseudos[2].to(self.device))
        x = self.activation_layer(x)
        
        x3= self.pool3(x)
        
        ######## ico-(data_ico-3) layer ########       
        x = self.conv4(x3,edge_indexes[3].to(self.device), pseudos[3].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv4_s(x,edge_indexes[3].to(self.device), pseudos[3].to(self.device))
        x = self.activation_layer(x)
        
        x4= self.pool4(x)
        
        ######## ico-(data_ico-4) layer ######## 
        x = self.conv5(x4,edge_indexes[4].to(self.device), pseudos[4].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv5_s(x,edge_indexes[4].to(self.device), pseudos[4].to(self.device))
        x = self.activation_layer(x)
        
        x5= self.pool5(x)
        
        ######## ico-(data_ico-5) layer ######## 
        x = self.conv6(x5,edge_indexes[5].to(self.device), pseudos[5].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv6_s(x,edge_indexes[5].to(self.device), pseudos[5].to(self.device))
        x = self.activation_layer(x)
                
        x = self.upsample2(x)
        
        ######### ico-(data_ico-4) layer ########
        x = torch.cat([x,x4], dim=1)
        
        x = self.conv7(x,edge_indexes[4].to(self.device),pseudos[4].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv7_s(x,edge_indexes[4].to(self.device), pseudos[4].to(self.device))
        x = self.activation_layer(x)
        
        x = self.upsample3(x)
        
        ######### ico-(data_ico-3) layer ########
        x = torch.cat([x,x3], dim=1)
        
        x = self.conv8(x,edge_indexes[3].to(self.device), pseudos[3].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv8_s(x,edge_indexes[3].to(self.device), pseudos[3].to(self.device))
        x = self.activation_layer(x)
        
        x = self.upsample4(x)
        
        ######### ico-(data_ico-2) layer ########       
        x = torch.cat([x,x2], dim = 1)
        
        x = self.conv9(x,edge_indexes[2].to(self.device), pseudos[2].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv9_s(x,edge_indexes[2].to(self.device), pseudos[2].to(self.device))
        x = self.activation_layer(x)
               
        x = self.upsample5(x)
        
        ######### ico-(data_ico-1) layer ########       
        x = torch.cat([x,x1], dim = 1)
        
        x = self.conv10(x,edge_indexes[1].to(self.device), pseudos[1].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv10_s(x,edge_indexes[1].to(self.device), pseudos[1].to(self.device))
        x = self.activation_layer(x)
        
        x = self.upsample6(x)
        
        ######### ico-data_ico layer ########      
        x = torch.cat([x,x_in], dim = 1)
        
        x = self.conv11(x,edge_input, pseudos[0].to(self.device))
        x = self.activation_layer(x)
        
        x = self.conv11_s(x,edge_input, pseudos[0].to(self.device))
        x = self.activation_layer(x)
        
        ####### feedforward network part #######
        
        x = self.conv_label(x)
        x = self.activation_layer_2(x)
        
        
        ''' 
        # Uncomment to add extra layer
        
        x = self.conv_label2(x)
        x = self.activation_layer_2(x)        
        '''        
        
        #####  downsample from data_ico level to control_ico level #####
        for i in range(self.level_dif):
            x= self.down_to_control[i](x)        
        
        def_idxs = self.softmax_layer(x) # new verticies locations
                
        warped_moving_img, deformed_control_ico, warps_moving = self.transformer_crf(moving_img, def_idxs)
        
        return warped_moving_img, deformed_control_ico, warps_moving

    