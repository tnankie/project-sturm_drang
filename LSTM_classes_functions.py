# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 12:03:48 2020

@author: work_cbdvl
"""
import torch
import numpy as np
class fc_net(torch.nn.Module):
    def __init__(self, n_features):
        super(fc_net, self).__init__()
        self.n_features = n_features
        self.n_hidden = 128 #512 original value
        self.taper = 5 #30 original value
        self.l_linear1 = torch.nn.Linear(self.n_features, self.n_hidden)
        self.l_linear2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.l_linear3 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.l_linear4 = torch.nn.Linear(self.n_hidden, self.taper)
        self.l_linear5 = torch.nn.Linear(self.taper, 1)
        
    def forward(self, x):
        m = torch.nn.ReLU()
        x = self.l_linear1(x)
        x = m(x)
        x = self.l_linear2(x)
        x = m(x)
        x = self.l_linear3(x)
        x = m(x)
        x = self.l_linear4(x)
        x = m(x)
        x = self.l_linear5(x)
        return x
        
        
        
    
class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 30 # number of hidden states orig 30
        self.n_layers = 2 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.drop1 = torch.nn.Dropout(0.5)
        self.l_linear1 = torch.nn.Linear(self.n_hidden*self.seq_len, 30)
        self.drop2 = torch.nn.Dropout(0.5)
        self.l_linear2 = torch.nn.Linear(30, 30)
        self.l_linear3 = torch.nn.Linear(30, 1)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        m = torch.nn.Sigmoid()  # was relu
        x = lstm_out.contiguous().view(batch_size,-1)
        x = self.drop1(x)
        x = self.l_linear1(x)
        x = self.drop2(x)
        x = m(x)
        x = self.l_linear2(x)
        x = m(x)
        return self.l_linear3(x)
    
class MV_LSTM3(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM3, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 128 # number of hidden states orig 30
        self.n_layers = 2 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.drop1 = torch.nn.Dropout(0.5)
        self.l_linear1 = torch.nn.Linear(self.n_hidden, 30) #changed in 3
        self.drop2 = torch.nn.Dropout(0.5)
        self.l_linear2 = torch.nn.Linear(30, 30)
        self.l_linear3 = torch.nn.Linear(30, 1)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        m = torch.nn.Sigmoid()  # was relu
        x = lstm_out.contiguous().view(-1, self.n_hidden) # changed in 3
        x = self.drop1(x)
        x = self.l_linear1(x)
        x = self.drop2(x)
        x = m(x)
        x = self.l_linear2(x)
        x = m(x)
        x =  self.l_linear3(x)
        x = x.view(batch_size, -1) # new for 3
        x = x[:, -1] # get last batch of labels, also new for 3
        return x
    
class MV_LSTM2(torch.nn.Module):
    def __init__(self, vocab_size, n_features, seq_length, drop_prob=0.5):
        super(MV_LSTM2, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 256 # number of hidden states orig 30
        self.n_layers = 2 # number of LSTM layers (stacked)
        self.embedding = torch.nn.Embedding(vocab_size, n_features)
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 dropout=drop_prob,
                                 batch_first = True, )
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.drop1 = torch.nn.Dropout(0.5)
        self.l_linear1 = torch.nn.Linear(self.n_hidden*self.seq_len, 30)
        self.drop2 = torch.nn.Dropout(0.5)
        self.l_linear2 = torch.nn.Linear(30, 30)
        self.l_linear3 = torch.nn.Linear(30, 1)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        self.hidden = (hidden_state, cell_state)
        return self.hidden
    
    
    def forward(self, x):
        try:        
            batch_size, seq_len, _ = x.shape
        except:
            batch_size, seq_len = x.shape
        x = x.long()
        embeds = self.embedding(x)
        print("Embedding shape: {}".format(embeds.shape))
        lstm_out, self.hidden = self.l_lstm(embeds, self.hidden)
        # b = self.hidden
        print("lstm_out shape: {}, hidden shape 1: {}, hidden shape 2: {}".format(lstm_out.shape, self.hidden[0].shape, self.hidden[1].shape))
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        m = torch.nn.Sigmoid()  # was relu
        x = lstm_out.contiguous().view(batch_size,-1)
        print("pre fc 1 dropout,  x shape: {}".format(x.shape))
        x = self.drop1(x)
        print("post fc 1 dropout,  x shape: {}".format(x.shape))
        x = self.l_linear1(x)
        print("pre fc 1 relu dropout,  x shape: {}".format(x.shape))
        x = self.drop2(x)
        print("pre fc 1 relu,  x shape: {}".format(x.shape))
        x = m(x)
        print("pre fc 2,  x shape: {}".format(x.shape))
        x = self.l_linear2(x)
        print("pre fc 2 relu,  x shape: {}".format(x.shape))
        x = m(x)
        print("pre fc 3,  x shape: {}".format(x.shape))
        x = self.l_linear3(x)
        print("post fc 3,  x shape: {}".format(x.shape))
        return x, self.hidden
    
    
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
    return np.array(X), np.array(y)


class Dataset(torch.utils.data.Dataset):
  # 'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        # 'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y
    

