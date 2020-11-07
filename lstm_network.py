# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:46:45 2020

@author: tnank
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sb
import sklearn
torch.manual_seed(42)

#%%

data = pd.read_csv("lstm_data.dat", index_col=0)

#%%
cuda = torch.device("cuda")

#%%
down_sample_id = np.arange(1, 5120, 10)

#%%
x = torch.FloatTensor(data.iloc[0:500,down_sample_id].values)#.cuda()

#%%
y = torch.FloatTensor(data.iloc[0:500,5121])#.cuda()

#%%

class sim_lstm(nn.Module):
    def __init__(self, input_size=1, output_size=1, hiden_1=256):
        super().__init__()
        self.hidden_layer = hiden_1
        self.lstm = nn.LSTM(input_size, hiden_1)
        self.linear = nn.Linear(hiden_1, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer), torch.zeros(1,1,self.hidden_layer))
        
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
#%%
model = sim_lstm()#.cuda()
loss_func = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

#%%
epochs = 10

for i in range(epochs):
    for j in np.arange(0, len(y)):
        optimiser.zero_grad()
        #model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer, device=cuda), torch.zeros(1, 1, model.hidden_layer, device=cuda))
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer), torch.zeros(1, 1, model.hidden_layer))
        y_pred = model(x[j])
        single_loss = loss_func(y_pred, y[j])
        single_loss.backward()
        optimiser.step()
        
        
    print("Epoch: {a:5} loss: {b:10.8f}".format(a = i, b = single_loss.item()))
