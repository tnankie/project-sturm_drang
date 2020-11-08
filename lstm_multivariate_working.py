
#%%
#%%
import random
import numpy as np
import torch

# multivariate data preparation
from numpy import array
from numpy import hstack
class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 30 # number of hidden states
        self.n_layers = 2 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
        
    
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
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)
    
#%%
import pandas as pd
#sample_data = np.load("abs_np.npy")
d1 = pd.read_csv("data\lstm_spectral_data.csv", index_col = 0)
d1 = d1.iloc[:,:-1]
d1 = d1.dropna()
#%%
tracks = {}
count = 0
for c,v in enumerate(d1["Track Section"].values):
    if v in tracks.keys():
        pass
    else:
        tracks[v] = count
        count += 1

#%%
d1 = d1.rename(columns={"Track Section": "trasec"})
d1["tr"] = d1.apply(lambda x: tracks[x.trasec], axis=1)
d1 = d1.drop("trasec", axis =1)
#%%
del tracks 
#%%
import random
import numpy as np
import torch

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
        seq_x, seq_y = sequences[i:end_ix, :-2], sequences[end_ix-1, -2]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
 
# # define input sequence
# in_seq1 = array([x for x in range(0,300,10)])
# in_seq2 = array([x for x in range(5,305,10)])
# out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# # convert to [rows, columns] structure
# in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# out_seq = out_seq.reshape((len(out_seq), 1))
# # horizontally stack columns
# dataset = hstack((in_seq1, in_seq2, out_seq))

#%%
n_features = 256 # this is number of parallel inputs
n_timesteps = 200 # this is number of timesteps

# convert dataset into input/output
Xx, yy = split_sequences(d1.iloc[:4000,:].values, n_timesteps)

print(Xx.shape, yy.shape)

# create NN
mv_net = MV_LSTM(n_features,n_timesteps)
mv_net.cuda()
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-1)
mv_net.cuda()
train_episodes = 1
batch_size = 512
#%%
xt= torch.from_numpy(Xx)
yt = torch.from_numpy(yy)
#%%
mv_net.train()
for t in range(train_episodes):
    for b in range(0,len(xt),batch_size):
        inpt = xt[b:b+batch_size,:,:]
        target = yt[b:b+batch_size]    
        
        x_batch = torch.tensor(inpt,dtype=torch.float32).cuda()    
        y_batch = torch.tensor(target,dtype=torch.float32).cuda()
        # print(x_batch.size(0))
        mv_net.init_hidden(x_batch.size(0))
    #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
    #    lstm_out.contiguous().view(x_batch.size(0),-1)
        output = mv_net(x_batch)
        # print("The output shape is: {}. The target is shape: {} \n ******".format(output.shape, y_batch.shape))
        t_loss = criterion(output.view(-1), y_batch.view(-1))  
        
        t_loss.backward()
        optimizer.step()        
        optimizer.zero_grad() 
    print('step : ' , t , 'loss : ' , t_loss.item())
    
# for b in range(0,len(Xx), batch_size):    
#     print("batch index:", b, "len X:", len(Xx))
#     print("X:", Xx[b:b+batch_size,:,:])
#     print("y:", yy[b:b+batch_size])