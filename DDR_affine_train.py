#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:07:29 2022

@author: Mohamed A. Suliman

email: mohamedabdall78@hotmail.com
"""

import numpy as np
import torch
import torch.nn as nn
from models.DDR_affine_model import DDR_affine
from models.DDR_dataloader import MRIImages


#################################
### Set your hyper-parameters ###
#################################

batch_size = 1
in_channels = 2 
out_channels = 1024
data_ico = 6
learning_rate = 1e-5
num_feat= [16, 16, 16, 16]

best_val = 1000
lambda_mse =1.0
lambda_cc  =10.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m_checkpoint = True
testing = True
save_rot_matrices = True 

Num_Epochs = 100

############################
### Set your directories ###
############################

moving_dir = 'moving_images/'  # moving imgs location
target_dir = 'target_images/'  # target imgs location

# moving imgs are named as: ID.L.sulc.ico6.shape.gii
# target img is named as: MSMSulc.L.sulc.ico6.shape.gii
# IDs are loaded into Id_files

# moving imgs Ids 
Id_file_t1 = 'DDR_files/Subjects_IDs/Subjects_ID_1'
Id_file_t2 = 'DDR_files/Subjects_IDs/Subjects_ID_2'
Id_file_t3 = 'DDR_files/Subjects_IDs/Subjects_ID_3'
 
Id_file_val = 'DDR_files/Subjects_IDs/Subjects_ID_val'
Id_file_test = 'DDR_files/Subjects_IDs/Subjects_ID_test' # if testing == True

moving_suffix = '.L.sulc.ico6.shape.gii'  # names without the Id number
target_prefix = 'MSMSulc'
target_suffix = '.L.sulc.ico6.shape.gii'

edge_in =torch.LongTensor(np.load('DDR_files/edge_index_'+str(data_ico)+'.npy')).to(device)

save_model_dir = 'results/models/'
save_rot_mat_dir = 'results/rot_matrices/'

### you probably won't need to change anything beyond this point ###

def CCLoss(x,y):
    CC_Loss = 1 - ((x - x.mean()) * (y - y.mean())).mean() / x.std() / y.std()
    return CC_Loss

## define the model ##        
model = DDR_affine(in_ch=in_channels, out_ch=out_channels, num_features=num_feat, 
                   data_ico=data_ico, device=device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=1e-7)

MSE_criterion = nn.MSELoss()
CC_criterion  = CCLoss

print("The device is '{}' ".format(device))
print("The DDR Affine Model has {} Paramerters".format(sum(x.numel() for x in model.parameters())))
  
## define datasets ##

train_dataset = MRIImages(moving_dir, 
                          target_dir, 
                          moving_suffix,
                          target_prefix,
                          target_suffix,
                          Id_file1=Id_file_t1)#, Id_file2=Id_file_t2, Id_file3=Id_file_t3)

val_dataset = MRIImages(moving_dir, 
                        target_dir, 
                        moving_suffix,
                        target_prefix,
                        target_suffix,
                        Id_file1=Id_file_val)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)

val_dataloader  = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                              shuffle=False, pin_memory=True)                                             

print('Num of Train Images = ',len(train_dataloader))  
print('Num of Val Images = ',len(val_dataloader))  
print('Num of Epochs =', Num_Epochs) 

## define validation and testing functions ##
def DDR_validation(dataloader,edge_in):    
    model.eval()
    
    val_losses_mse = torch.zeros((len(dataloader),1))
    val_losses_cc  = torch.zeros((len(dataloader),1))
    
    for batch_idx, (moving_ims, target_ims) in enumerate(dataloader):
        moving_ims, target_ims = (moving_ims.squeeze(0)).to(device), (target_ims.squeeze(0)).to(device)
        
        with torch.no_grad():
            affined_moving_img_val, _ = model(moving_ims,target_ims, edge_in)
            
        val_losses_mse[batch_idx,:] = MSE_criterion(affined_moving_img_val, target_ims).to('cpu')
        val_losses_cc[batch_idx,:]  = CC_criterion(affined_moving_img_val, target_ims).to('cpu')

    return val_losses_mse+val_losses_cc, torch.mean(val_losses_mse, axis=0), torch.mean(val_losses_cc, axis=0)


def DDR_testing(dataloader,edge_in):
    model.eval()
         
    test_rot_matrices = torch.zeros(16,len(dataloader))
    val_losses_mse = torch.zeros((len(dataloader),1))
    val_losses_gcc = torch.zeros((len(dataloader),1))
    
    for batch_idx, (moving_ims, target_ims) in enumerate(dataloader):
        
        moving_ims, target_ims = (moving_ims.squeeze(0)).to(device), (target_ims.squeeze(0)).to(device)
                        
        with torch.no_grad():
            affined_moving_img_test, rotation_mat_test = model(moving_ims,target_ims,edge_in) 
            val_losses_mse[batch_idx,:] = MSE_criterion(affined_moving_img_test, target_ims).to('cpu')
            val_losses_gcc[batch_idx,:] = CC_criterion(affined_moving_img_test, target_ims).to('cpu')
            test_rot_matrices[:,batch_idx] = (rotation_mat_test.reshape(-1,1)).squeeze()
    
    return val_losses_mse, val_losses_gcc, test_rot_matrices


def print_during_training(epoch,train_loss_sum,train_loss_mse,train_loss_cc,
                          val_loss_mean,val_loss_mse,val_loss_cc):
    print('\n')
    print('(Ep = {}) ** (T.L = {:.5}) ** (T.MSE = {:.5}) ** (T.CC Accu = {:.5})\n ********** (V.L = {:.5}) ** (V.MSE = {:.5}) ** (V.CC Accu = {:.5})'
            .format(epoch, train_loss_sum[epoch].numpy()[0],
                    train_loss_mse[epoch].numpy()[0], 1.0-train_loss_cc[epoch].numpy()[0],
                    val_loss_mean[epoch].numpy()[0],val_loss_mse.numpy()[0],1.0-val_loss_cc.numpy()[0]) )   
  
    
train_loss_mse = torch.zeros(Num_Epochs,1)
train_loss_cc  = torch.zeros(Num_Epochs,1)
train_loss_sum = torch.zeros(Num_Epochs,1)
val_loss_mean  = torch.zeros(Num_Epochs,1)
num_train_data = len(train_dataloader)

### Start traning process ### 
    
for epoch in range(Num_Epochs):

    running_losses_mse = 0
    running_losses_cc  = 0
    running_losses_sum = 0
    
    for batch_idx, (moving_ims, target_ims) in enumerate(train_dataloader):
        model.train()

        moving_ims, target_ims = (moving_ims.squeeze(0)).to(device), (target_ims.squeeze(0)).to(device)
        
        optimizer.zero_grad()         
        
        affined_moving_img, _ = model(moving_ims,target_ims, edge_in)

        loss_mse = MSE_criterion(affined_moving_img, target_ims)
        loss_cc  = CC_criterion(affined_moving_img, target_ims)
        loss = lambda_mse*loss_mse+lambda_cc*loss_cc

        loss.backward()
        optimizer.step() 
        
        running_losses_sum+=loss.item()
        running_losses_mse+=loss_mse.item()
        running_losses_cc+=loss_cc.item()
        
    train_loss_sum[epoch] = torch.tensor(running_losses_sum/num_train_data)   
    train_loss_cc[epoch]  = torch.tensor(running_losses_cc/num_train_data)        
    train_loss_mse[epoch] = torch.tensor(running_losses_mse/num_train_data)
    
    ## Start validation ####
    
    val_loss, val_loss_mse,val_loss_cc = DDR_validation(val_dataloader, edge_in)
    val_loss_mean[epoch] = torch.mean(val_loss, axis=0)
    
    if (epoch+1)%1 ==0:
        print_during_training(epoch,train_loss_sum,train_loss_mse,train_loss_cc,
                              val_loss_mean,val_loss_mse,val_loss_cc)
    
    scheduler.step(val_loss_mean[epoch])
    
    ## save the model ? ##
    if m_checkpoint:
        
        torch.save(model.state_dict(), save_model_dir+'trained_affine_model.pkl') 
    
        ### save the model if it works better on the val set ###
        if val_loss_mean[epoch].numpy()[0] < best_val:
            best_val = val_loss_mean[epoch].numpy()[0] 
            torch.save(model.state_dict(), save_model_dir+'best_affine_model.pkl')
            print('-- New best model saved --')
        
print('Done from training.')

### Start testing ### 

if testing:
        
    print('\nStart testing...')
    
    test_dataset = MRIImages(moving_dir, target_dir, moving_suffix, target_prefix,
                             target_suffix, Id_file1=Id_file_test)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, pin_memory=True)
    
    print('Num of Test Images = ',len(test_dataloader))
    
    test_loss_mes, test_loss_gcc, test_rot_matrices = DDR_testing(test_dataloader,edge_in)
    test_mse = torch.mean(test_loss_mes, axis=0)
    test_cc = torch.mean(test_loss_gcc, axis=0)
    
    print('\n### Test Results ###')
    print('Test loss in MSE = {:.4}'.format(test_mse.numpy()[0]))
    print('Test accu in CC = {:.4}'.format(1-test_cc.numpy()[0]))
    
    if save_rot_matrices:
        print('\nSaving rotation matrices...')
        
        test_subj_ids = open(Id_file_test, "r").read().splitlines()
        
        for idx , subj_id in enumerate(test_subj_ids):
            
            rot_matrix = (test_rot_matrices[:,idx].reshape(4,4)).T 
            
            np.savetxt(save_rot_mat_dir+subj_id, rot_matrix, fmt='%.5f')
                            
        print('Rotation matrices saved!')
    
