# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 12:12:15 2020

@author: work_cbdvl
"""
import torch
from LSTM_classes_functions import Dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

import pandas as pd
d1 = pd.read_csv("data\lstm_spectral_data.csv", index_col = 0)
d1 = d1.iloc[:,:-1]
d1 = d1.dropna()
cols = d1.columns
#%%
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler().fit(d1)
d1 = scaler.transform(d1)
d1 = pd.DataFrame(d1, columns = cols)
tracks = {}
count = 0
for c,v in enumerate(d1["Track Section"].values):
    if v in tracks.keys():
        pass
    else:
        tracks[v] = count
        count += 1

d1 = d1.rename(columns={"Track Section": "trasec"})
d1["tr"] = d1.apply(lambda x: tracks[x.trasec], axis=1)
d1 = d1.drop("trasec", axis =1)

del tracks 

#%%
import random
import numpy as np
import torch
import torch.nn as nn

# multivariate data preparation
from numpy import array
from numpy import hstack
 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
#%%
n_timesteps = 40 # this is number of timesteps was 200

# # convert dataset into input/output
# train_x, train_y = np.float32(d1.iloc[:69133,:-2].values), np.float32(d1.iloc[:69133,-2].values)
# test_x, test_y = np.float32(d1.iloc[69133:,:-2].values), np.float32(d1.iloc[69133:,-2].values)
# del d1
train_x, train_y = split_sequences(np.float32(d1.iloc[:69133,:-1].values), n_timesteps)
test_x, test_y = split_sequences(np.float32(d1.iloc[69133:,:-1].values), n_timesteps)

# print(Xx.shape, yy.shape)
#%%
import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 1024

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

#%%
# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)


#%%
# create NN
n_features = 256 # this is number of parallel inputs

train_episodes = 20 # this is the number of epochs
clip = 5 # gradient clipping
from LSTM_classes_functions import MV_LSTM
mv_net = MV_LSTM(n_features,n_timesteps)
if use_cuda:
    mv_net.cuda()
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-1)
if use_cuda:
    mv_net.cuda()
print(mv_net)

#%%
mv_net.train()
for t in range(train_episodes):
    for inputs, labels in train_loader:
        mv_net.train()
        
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
    
        mv_net.init_hidden(inputs.shape[0])
    #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
    #    lstm_out.contiguous().view(x_batch.size(0),-1)
        output = mv_net(inputs)
        # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, y_batch.shape))
        t_loss = criterion(output.view(-1), labels.view(-1))  
        
        t_loss.backward()
        nn.utils.clip_grad_norm_(mv_net.parameters(), clip)
        optimizer.step()        
        optimizer.zero_grad() 
    print('step : ' , t , 'train loss : ' , t_loss.item())
    for test_in, test_lab in test_loader:
        mv_net.init_hidden(test_in.shape[0])
        mv_net.eval()
        if use_cuda:
            test_in, test_lab = test_in.cuda(), test_lab.cuda()
        test_out = mv_net(test_in)
        v_loss = criterion(test_out.view(-1), test_lab.view(-1))
    print('step : ' , t , 'test loss : ' , v_loss.item())
#%%
mv_net.train()
# for t in range(train_episodes):
#     for b in range(0,len(xt),batch_size):
#         inpt = xt[b:b+batch_size,:,:]
#         target = yt[b:b+batch_size]    
        
#         x_batch = torch.tensor(inpt,dtype=torch.float32).cuda()    
#         y_batch = torch.tensor(target,dtype=torch.float32).cuda()
#         # print(x_batch.size(0))
#         mv_net.init_hidden(x_batch.size(0))
#     #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
#     #    lstm_out.contiguous().view(x_batch.size(0),-1)
#         output = mv_net(x_batch)
#         # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, y_batch.shape))
#         t_loss = criterion(output.view(-1), y_batch.view(-1))  
        
#         t_loss.backward()
#         optimizer.step()        
#         optimizer.zero_grad() 
#     print('step : ' , t , 'loss : ' , t_loss.item())
    
# # for b in range(0,len(Xx), batch_size):    
# #     print("batch index:", b, "len X:", len(Xx))
# #     print("X:", Xx[b:b+batch_size,:,:])
# #     print("y:", yy[b:b+batch_size])




# # Datasets
# partition = # IDs
# labels = # Labels

# # Generators
# training_set = Dataset(partition['train'], labels)
# training_generator = torch.utils.data.DataLoader(training_set, **params)

# validation_set = Dataset(partition['validation'], labels)
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# # Loop over epochs
# for epoch in range(max_epochs):
#     # Training
#     for local_batch, local_labels in training_generator:
#         # Transfer to GPU
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#         # Model computations
#         [...]

#     # Validation
#     with torch.set_grad_enabled(False):
#         for local_batch, local_labels in validation_generator:
#             # Transfer to GPU
#             local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#             # Model computations
#             [...]