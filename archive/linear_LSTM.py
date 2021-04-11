# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:53:14 2020

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

data = pd.read_csv("linear_lstm_data.csv", index_col=0)
#%%
cuda = torch.device("cuda")

#%%
class sim_lstm(nn.Module):
    def __init__(self, inp_siz=64, opu_siz=2, hidd=256, num_lay = 2, drp_o = 0.5, lin_hid = 256):
        super().__init__()
        self.hidden_layer = hidd
        self.lstm = nn.LSTM(input_size = inp_siz, hidden_size = hidd, num_layers= num_lay, dropout= drp_o)
        self.linear = nn.Linear(hidd, opu_siz)
        self.hidden_cell = (torch.zeros(num_lay, 1, self.hidden_layer), torch.zeros(num_lay, 1, self.hidden_layer))
        
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions
#%%
model1 = sim_lstm()
print(model1)

#%%
train_window = 256
def createinpout_seq(inp_d, tw):
    inout_seq = []
    a = np.zeros((len(inp_d) - tw, tw + 2))
    L = len(inp_d)
    for c,v in enumerate(range(L-tw)):
        train_seq = inp_d[v:v+tw, 0]
        train_lab = inp_d[v+tw, 1:]
        inout_seq.append((train_seq, train_lab))
        a[c,0:256] = train_seq
        a[c,256:] = train_lab
        
    return inout_seq, a
#%%
track_dic = {}
count = 1
for i in data["Track Section"]:
    if i in track_dic:
        pass
    else:
        track_dic[i] = count
        count += 1
#%%
data["pos"] = data.apply(lambda x: track_dic[x["Track Section"]], axis =1)
#%%

sequence = np.arange(20000000, 20000000 + 5120*180, 100)
#%%
test_data = data.iloc[sequence, [0,1,4]].values
del data
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
test_norm = scaler.fit_transform(test_data)
test_norm_t = torch.FloatTensor(test_norm).cuda()
print(test_norm[:5])
print(test_norm[-5:])
print(test_norm_t[:5])
print(test_norm_t[-5:])
#%%
test_seq, np_a =  createinpout_seq(test_norm, train_window)
np_a_t = torch.from_numpy(np_a).cuda()
#%%
from torch.utils.data import TensorDataset, DataLoader
#%%
aa = np_a_t[:,:256]
bb = np_a_t[:,256:]
train_data =TensorDataset(aa, bb)
b_size = 32
train_loader = DataLoader(train_data, shuffle=True, batch_size = b_size)

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
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidd_lay_si=100, outp_si=2, n_lay=2, btch = 1):
        super().__init__()
        self.hidden_layer_size = hidd_lay_si

        self.lstm = nn.LSTM(input_size, hidd_lay_si, num_layers = n_lay)

        self.lin1 = nn.Linear(hidd_lay_si, hidd_lay_si)
        self.lin2 = nn.Linear(hidd_lay_si, hidd_lay_si)
        self.lin3 = nn.Linear(hidd_lay_si, hidd_lay_si)
        
        self.lin4 = nn.Linear(hidd_lay_si, outp_si)

        self.hidden_cell = (torch.zeros(n_lay, btch, self.hidden_layer_size, device=cuda),
                            torch.zeros(n_lay, btch, self.hidden_layer_size, device=cuda))
        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()
        self.batch = btch

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) , self.batch, -1), self.hidden_cell)
        out = self.lin1(lstm_out.view(len(input_seq), -1))
        
        out = self.dropout(out)
        out = self.sig(out)
        out = self.lin2(out)
        out = self.dropout(out)
        out = self.sig(out)
        out = self.lin3(out)
        out = self.sig(out)
        predictions = self.lin4(out)
        return predictions[-1]
    
#%%
model2 = LSTM(btch = b_size)
model2.cuda()
loss_function = nn.MSELoss()
optim = torch.optim.Adam(model2.parameters(), lr=0.001)
print(model2)
#%%
model2.cuda()

#%%
epochs = 5

for i in range(epochs):
    for seq, labels in t2:
        optim.zero_grad()
        print("optimized grads")
        model2.hidden_cell = (torch.zeros(2, 1, model2.hidden_layer_size, device=cuda),
                        torch.zeros(2, 1, model2.hidden_layer_size, device=cuda))

        y_pred = model2(seq)
        print("predicted")

        single_loss = loss_function(y_pred, labels)
        print("loss")
        single_loss.backward()
        print("backward")
        optim.step()
        print("step")

    if i%2 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

#%%
train_on_gpu = True
# training params
# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=lr)

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    model2.cuda()

model2.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = model2.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        model2.zero_grad()

        # get the output from the model
        output, h = model2(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model2.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
