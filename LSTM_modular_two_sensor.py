# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 12:12:15 2020

@author: work_cbdvl
"""
import torch
import seaborn as sns
from LSTM_classes_functions import Dataset
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True



import pandas as pd
d1 = pd.read_csv(".\data\lstm_spectral_data_combined.csv", index_col = 0)
d1 = d1.iloc[:,:-1]
d1 = d1.dropna()
cols = d1.columns


from sklearn.preprocessing import RobustScaler
dy = d1.iloc[:,-2:]
dd = np.log(d1.iloc[:,:-2]+1)
# dd = d1.iloc[:,:-2]
scaler = RobustScaler().fit(dd)
d1 = scaler.transform(dd)
d1 = pd.DataFrame(d1, columns = cols[:-2])
d1["vel"] = dy["Actual Velocity"].values
d1["Track Section"] = dy["Track Section"].values

del dd
del dy


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

d1 = d1.loc[d1.vel < 90]

#%%
import matplotlib.pyplot as plt

#%%
# Parameters
n_timesteps = 16 # this is number of timesteps was 200
batch_size = 2 ** 9 # training batch size
n_features = 512 # this is number of parallel inputs
lstm_hidden = 150 # lstm hidden dimension size
lstm_layers = 5 # number of lstm layers
fc_hidden = 1 # size of hidden layers in fc network
train_episodes = 60 # this is the number of epochs
clip =  1 # gradient clipping
amd_lr = 1e-3


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



# # convert dataset into input/output
# train_x, train_y = np.float32(d1.iloc[:69133,:-2].values), np.float32(d1.iloc[:69133,-2].values)
# test_x, test_y = np.float32(d1.iloc[69133:,:-2].values), np.float32(d1.iloc[69133:,-2].values)
# del d1
train_x, train_y = split_sequences(np.float32(d1.iloc[:69133,:-1].values), n_timesteps)
test_x, test_y = split_sequences(np.float32(d1.iloc[69133:,:-1].values), n_timesteps)

# print(Xx.shape, yy.shape)

import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders


# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

# print('Sample input size: ', sample_x.size()) # batch_size, seq_length
# print('Sample input: \n', sample_x)
# print()
# print('Sample label size: ', sample_y.size()) # batch_size
# print('Sample label: \n', sample_y)



# create NN

from LSTM_classes_functions import MV_LSTM4
mv_net = MV_LSTM4(n_features, n_timesteps, lstm_hidden, lstm_layers, fc_hidden)
# from LSTM_classes_functions import fc_net
# mv_net = fc_net(n_features, n_timesteps, lstm_hidden, lstm_layers, fc_hidden)
if use_cuda:
    mv_net.cuda()
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=amd_lr)
if use_cuda:
    mv_net.cuda()
print(mv_net)


mv_net.train()
t_losses = []
v_losses = []
b_losses = []
for t in range(train_episodes):
    counter = 0
    for inputs, labels in train_loader:
        mv_net.train()
        
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
    
        mv_net.init_hidden(inputs.shape[0])
    #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
    #    lstm_out.contiguous().view(x_batch.size(0),-1)
        output = mv_net(inputs)
        # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, labels.shape))
        t_loss = criterion(output.view(-1), labels.view(-1))  
        # print('counter : ' , counter , 'train loss : ' , t_loss.item())
        b_losses.append(t_loss.item())
        t_loss.backward()
        nn.utils.clip_grad_norm_(mv_net.parameters(), clip)
        optimizer.step()        
        optimizer.zero_grad()
        counter += 1
    print('step : ' , t , 'train loss : ' , t_loss.item())
    t_losses.append(t_loss.item())
    
    for test_in, test_lab in test_loader:
        mv_net.init_hidden(test_in.shape[0])
        mv_net.eval()
        if use_cuda:
            test_in, test_lab = test_in.cuda(), test_lab.cuda()
        test_out = mv_net(test_in)
        v_loss = criterion(test_out.view(-1), test_lab.view(-1))
    print('step : ' , t , 'test loss : ' , v_loss.item())
    v_losses.append(v_loss.item())



q= 5 #first batch score to plot
x_val = np.arange(0,len(v_losses))
xb_val = np.arange(0,len(b_losses))/(len(b_losses)/len(v_losses))

title = "class: {}, batchsize: {}, timesteps: {}, first hidden layer: {}".format(type(mv_net), batch_size, n_timesteps, mv_net.n_hidden )
sns.scatterplot(x=x_val, y=v_losses, color= "r")
sns.scatterplot(x=xb_val[q:], y=b_losses[q:], color= "g")
sns.scatterplot(x=x_val, y=t_losses, color= "b").set_title(title)
plt.show()
batch_size2 = 20000
test_loader2 = DataLoader(test_data, shuffle=False, batch_size=batch_size2)
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
predict = []
val_losses = []
mv_net.eval()


for inputs, labels in test_loader2:
        
        
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
    
        mv_net.init_hidden(inputs.shape[0])
        output = mv_net(inputs)
        # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, y_batch.shape))
        t_loss = criterion(output.view(-1), labels.view(-1))
        val_losses.append(t_loss.item())
      
        predict.append(output.detach().to("cpu"))

a = np.zeros((n_timesteps -1, 1))
#b = np.zeros((7,1))



for i in np.arange(0,len(predict)):
    print(i)
    a = np.concatenate((a, predict[i].view(-1,1).numpy()), axis=0)
    # a = np.concatenate((a, b), axis=0)

d2 = d1.iloc[69133:,:]
d2["pred_lstm"] = a
b = d2.iloc[n_timesteps -1,-1]

d2.iloc[0:n_timesteps -2,-1] = b


import matplotlib.pyplot as plt
sns.displot(data = d2, x="pred_lstm", kind = "hist")
sns.displot(data = d2, x="vel", kind = "hist")
sns.displot(data = d2, x="pred_lstm", kind = "ecdf")
sns.displot(data = d2, x="vel", kind = "ecdf")
plt.show()
sns.scatterplot(data=d2, x="vel", y = "pred_lstm", alpha=0.05, s=2)
plt.show()
x_vals = np.arange(0, len(d2))
sns.scatterplot(x=x_vals, y=d2.vel, color = "r")
sns.scatterplot(x=x_vals, y=d2.pred_lstm, color = "b")      
plt.show()
mid = int(len(x_vals)/5)

x_vals = np.arange(0, len(d2))
sns.scatterplot(x=x_vals[2 * mid :3 * mid], y=d2.vel.iloc[2 * mid :3 * mid], color = "r")
sns.scatterplot(x=x_vals[2 * mid :3 * mid], y=d2.pred_lstm.iloc[2 * mid :3 * mid], color = "b")      
plt.show()

train_xp, train_yp = split_sequences(np.float32(d1.iloc[5000:15000,:-1].values), n_timesteps)
train_datap = TensorDataset(torch.from_numpy(train_xp), torch.from_numpy(train_yp))
test_loader3 = DataLoader(train_datap, shuffle=False, batch_size=batch_size2)
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
predict2 = []
val_losses2 = []
mv_net.eval()


for inputs, labels in test_loader3:
        
        
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
    
        mv_net.init_hidden(inputs.shape[0])
        output = mv_net(inputs)
        # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, y_batch.shape))
        t_loss = criterion(output.view(-1), labels.view(-1))
        val_losses2.append(t_loss.item())
      
        predict2.append(output.detach().to("cpu"))

aa = np.zeros((n_timesteps -1, 1))
#b = np.zeros((7,1))


for i in np.arange(0,len(predict2)):
    print(i)
    aa = np.concatenate((aa, predict2[i].view(-1,1).numpy()), axis=0)
    # a = np.concatenate((a, b), axis=0)

d3 = d1.iloc[5000:15000,:]
d3["pred_lstm"] = aa
bb = d3.iloc[n_timesteps -1,-1]

d3.iloc[0:n_timesteps -2,-1] = bb


import matplotlib.pyplot as plt
sns.displot(data = d3, x="pred_lstm", kind = "hist")
sns.displot(data = d3, x="vel", kind = "hist")
sns.displot(data = d3, x="pred_lstm", kind = "ecdf")
sns.displot(data = d3, x="vel", kind = "ecdf")
plt.show()
sns.scatterplot(data=d3, x="vel", y = "pred_lstm", alpha=0.05, s=2)
plt.show()
x_vals = np.arange(0, len(d3))
sns.scatterplot(x=x_vals, y=d3.vel, color = "c")
sns.scatterplot(x=x_vals, y=d3.pred_lstm, color = "m")      
plt.show()
mid = int(len(x_vals)/5)

x_vals = np.arange(0, len(d3))
sns.scatterplot(x=x_vals[2 * mid :3 * mid], y=d3.vel.iloc[2 * mid :3 * mid], color = "c")
sns.scatterplot(x=x_vals[2 * mid :3 * mid], y=d3.pred_lstm.iloc[2 * mid :3 * mid], color = "m")      
plt.show()

#%%
# # create NN
# n_features = 256 # this is number of parallel inputs

# train_episodes = 15 # this is the number of epochs
# clip = 5 # gradient clipping
# from LSTM_classes_functions import MV_LSTM3
# mv_net = MV_LSTM3(n_features,n_timesteps)
# if use_cuda:
#     mv_net.cuda()
# criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
# optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-2)
# if use_cuda:
#     mv_net.cuda()
# print(mv_net)

# #%%
# mv_net.train()
# t_losses = []
# v_losses = []
# for t in range(train_episodes):
#     counter = 0
#     for inputs, labels in train_loader:
#         mv_net.train()
        
#         if use_cuda:
#             inputs, labels = inputs.cuda(), labels.cuda()
        
    
#         mv_net.init_hidden(inputs.shape[0])
#     #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
#     #    lstm_out.contiguous().view(x_batch.size(0),-1)
#         output = mv_net(inputs)
#         # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, y_batch.shape))
#         t_loss = criterion(output.view(-1), labels.view(-1))  
#         print('counter : ' , counter , 'train loss : ' , t_loss.item())
#         t_loss.backward()
#         nn.utils.clip_grad_norm_(mv_net.parameters(), clip)
#         optimizer.step()        
#         optimizer.zero_grad()
#         counter += 1
#     print('step : ' , t , 'train loss : ' , t_loss.item())
#     t_losses.append(t_loss.item())
    
#     for test_in, test_lab in test_loader:
#         mv_net.init_hidden(test_in.shape[0])
#         mv_net.eval()
#         if use_cuda:
#             test_in, test_lab = test_in.cuda(), test_lab.cuda()
#         test_out = mv_net(test_in)
#         v_loss = criterion(test_out.view(-1), test_lab.view(-1))
#     print('step : ' , t , 'test loss : ' , v_loss.item())
#     v_losses.append(v_loss.item())


# #%%
# x_val = np.arange(0,len(v_losses))
# title = "class: {}, batchsize: {}, timesteps: {}, first hidden layer: {}".format(type(mv_net), batch_size, n_timesteps, mv_net.n_hidden )
# sns.scatterplot(x=x_val, y=v_losses, color= "r")
# sns.scatterplot(x=x_val, y=t_losses, color= "b").set_title(title)








# #%%
# n_timesteps2 = 1 # this is number of timesteps was 200

# # # convert dataset into input/output
# # train_x, train_y = np.float32(d1.iloc[:69133,:-2].values), np.float32(d1.iloc[:69133,-2].values)
# # test_x, test_y = np.float32(d1.iloc[69133:,:-2].values), np.float32(d1.iloc[69133:,-2].values)
# # del d1
# train_x, train_y = split_sequences(np.float32(d1.iloc[:69133,:-1].values), n_timesteps2)
# test_x, test_y = split_sequences(np.float32(d1.iloc[69133:,:-1].values), n_timesteps2)

# # print(Xx.shape, yy.shape)
# #%%
# import torch
# from torch.utils.data import TensorDataset, DataLoader

# # create Tensor datasets
# train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

# test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# # dataloaders
# batch_size2 = 90000

# # make sure the SHUFFLE your training data
# train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size2)

# test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size2)

# #%%
# # obtain one batch of training data
# dataiter = iter(train_loader)
# sample_x, sample_y = dataiter.next()

# print('Sample input size: ', sample_x.size()) # batch_size, seq_length
# print('Sample input: \n', sample_x)
# print()
# print('Sample label size: ', sample_y.size()) # batch_size
# print('Sample label: \n', sample_y)


# #%%
# # create NN
# n_features = 256 # this is number of parallel inputs

# train_episodes = 300 # this is the number of epochs
# clip = 10 # gradient clipping
# from LSTM_classes_functions import fc_net
# mv_net2 = fc_net(n_features)
# if use_cuda:
#     mv_net2.cuda()
# criterion2 = torch.nn.MSELoss() # reduction='sum' created huge loss value
# optimizer2 = torch.optim.Adam(mv_net2.parameters(), lr=1e-3)
# if use_cuda:
#     mv_net2.cuda()
# print(mv_net2)

# #%%
# mv_net2.train()
# t_losses2 = []
# v_losses2 = []
# tb_losses2 = []
# for t in range(train_episodes):
#     mv_net2.train()
#     for inputs, labels in train_loader:
        
        
#         if use_cuda:
#             inputs, labels = inputs.cuda(), labels.cuda()
        
    
      
#         output = mv_net2(inputs)
#         # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, y_batch.shape))
#         t_loss2 = criterion2(output.view(-1), labels.view(-1))
#         tb_losses2.append(t_loss2.item())
#         optimizer2.zero_grad() #?????
#         t_loss2.backward()
#         # nn.utils.clip_grad_norm_(mv_net2.parameters(), clip)
#         optimizer2.step()        
        
#     # print('step : ' , t , 'train loss : ' , t_loss2.item())
#     t_losses2.append(t_loss2.item())
    
#     mv_net2.eval()
#     for test_in, test_lab in test_loader:
        
        
#         if use_cuda:
#             test_in, test_lab = test_in.cuda(), test_lab.cuda()
#         test_out = mv_net2(test_in)
#         v_loss = criterion2(test_out.view(-1), test_lab.view(-1))
#     # print('step : ' , t , 'test loss : ' , v_loss.item())
#     v_losses2.append(v_loss.item())


# x_val = np.arange(0,len(v_losses2))
# xb_val2 = np.arange(0,len(tb_losses2))/(len(train_loader))
# title = "class: {}, batchsize: {}, timesteps: {}, first hidden layer: {}".format(type(mv_net2), batch_size2, n_timesteps2, mv_net2.n_hidden )
# # sns.scatterplot(x=x_val, y=v_losses2, color= "r")
# # sns.scatterplot(x=xb_val2, y=tb_losses2, color= "g")
# # sns.scatterplot(x=x_val, y=t_losses2, color= "b").set_title(title)
# check = pd.DataFrame({"train_loss": t_losses2, "test_loss":v_losses2})
# sns.scatterplot(data=check).set_title(title)


# #%%
# # create NN
# n_features = 256 # this is number of parallel inputs

# train_episodes = 500 # this is the number of epochs
# clip = 10 # gradient clipping
# from LSTM_classes_functions import fc_net
# mv_net3 = fc_net(n_features)
# if use_cuda:
#     mv_net3.cuda()
# criterion3 = torch.nn.MSELoss() # reduction='sum' created huge loss value
# optimizer3 = torch.optim.Adam(mv_net3.parameters(), lr=1e-3)
# if use_cuda:
#     mv_net3.cuda()
# print(mv_net3)

# #%%
# mv_net3.train()
# t_losses3 = []
# v_losses3 = []
# tb_losses3 = []
# for t in range(train_episodes):
#     mv_net3.train()
        
        
#     if use_cuda:
#         inputs, labels = torch.from_numpy(np.float32(d1.iloc[:69133,:-2].values)).cuda(), torch.from_numpy(np.float32(d1.iloc[:69133,-2].values)).cuda()
        
    
      
#     output = mv_net3(inputs)
#         # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, y_batch.shape))
#     t_loss3 = criterion3(output.view(-1), labels.view(-1))
#     tb_losses3.append(t_loss3.item())
#     optimizer3.zero_grad() #?????
#     t_loss3.backward()
#         # nn.utils.clip_grad_norm_(mv_net2.parameters(), clip)
#     optimizer3.step()        
        
#     # print('step : ' , t , 'train loss : ' , t_loss2.item())
#     t_losses3.append(t_loss3.item())
    
#     mv_net3.eval()
 
        
#     if use_cuda:
#          test_in, test_lab = torch.from_numpy(np.float32(d1.iloc[69133:,:-2].values)).cuda(), torch.from_numpy(np.float32(d1.iloc[69133:,-2].values)).cuda()
#     test_out = mv_net3(test_in)
#     v_loss = criterion3(test_out.view(-1), test_lab.view(-1))
#     # print('step : ' , t , 'test loss : ' , v_loss.item())
#     v_losses3.append(v_loss.item())


# x_val = np.arange(0,len(v_losses3))
# xb_val3 = np.arange(0,len(tb_losses2))/(len(train_loader))
# title = "class: {}, batchsize: {}, timesteps: {}, first hidden layer: {}".format(type(mv_net3), batch_size2, n_timesteps2, mv_net3.n_hidden )
# # sns.scatterplot(x=x_val, y=v_losses2, color= "r")
# # sns.scatterplot(x=xb_val2, y=tb_losses2, color= "g")
# # sns.scatterplot(x=x_val, y=t_losses2, color= "b").set_title(title)
# check = pd.DataFrame({"train_loss": t_losses3, "test_loss":v_losses3})
# sns.scatterplot(data=check).set_title(title)
# #%%
# mv_net.train()
# # for t in range(train_episodes):
# #     for b in range(0,len(xt),batch_size):
# #         inpt = xt[b:b+batch_size,:,:]
# #         target = yt[b:b+batch_size]    
        
# #         x_batch = torch.tensor(inpt,dtype=torch.float32).cuda()    
# #         y_batch = torch.tensor(target,dtype=torch.float32).cuda()
# #         # print(x_batch.size(0))
# #         mv_net.init_hidden(x_batch.size(0))
# #     #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
# #     #    lstm_out.contiguous().view(x_batch.size(0),-1)
# #         output = mv_net(x_batch)
# #         # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, y_batch.shape))
# #         t_loss = criterion(output.view(-1), y_batch.view(-1))  
        
# #         t_loss.backward()
# #         optimizer.step()        
# #         optimizer.zero_grad() 
# #     print('step : ' , t , 'loss : ' , t_loss.item())
    
# # # for b in range(0,len(Xx), batch_size):    
# # #     print("batch index:", b, "len X:", len(Xx))
# # #     print("X:", Xx[b:b+batch_size,:,:])
# # #     print("y:", yy[b:b+batch_size])




# # # Datasets
# # partition = # IDs
# # labels = # Labels

# # # Generators
# # training_set = Dataset(partition['train'], labels)
# # training_generator = torch.utils.data.DataLoader(training_set, **params)

# # validation_set = Dataset(partition['validation'], labels)
# # validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# # # Loop over epochs
# # for epoch in range(max_epochs):
# #     # Training
# #     for local_batch, local_labels in training_generator:
# #         # Transfer to GPU
# #         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

# #         # Model computations
# #         [...]

# #     # Validation
# #     with torch.set_grad_enabled(False):
# #         for local_batch, local_labels in validation_generator:
# #             # Transfer to GPU
# #             local_batch, local_labels = local_batch.to(device), local_labels.to(device)

# #             # Model computations
# #             [...]


#%%

plot corelation of velocity and frequency (find the better predicting frequencies)
# min max scaler
then look at pure accelerometer data down down sampled by 1/3
discrete time slices 0-10 11-20 etc Not 0-10 1-11 
look at batch size again

look at providing labels for all time steps in sequence length
