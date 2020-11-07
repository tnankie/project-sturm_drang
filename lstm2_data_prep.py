# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:26:56 2020

@author: work_cbdvl
"""
import torch
import torch.nn as nn
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

target = pd.read_csv("snoops.csv")

data = pd.read_csv("./03-05-19-V118/03-05-19_ch1.csv", header=15)

# first problem - sychronise the data and target
# step 1 build date time features for both datasets
head = []
with open("./03-05-19-V118/03-05-19_ch1.csv", "r") as file:
    count = 0
    for line in file:
        if count > 16:
            break
        head.append(line)
        count += 1
    file.close()

# creating a datetime object from existing time and date features
#then filtering the dataset
target["datetime"] = target.apply(lambda x: str(x.Date) + " " + str(x.Time), axis = 1)
target["fixed"] = pd.to_datetime(target.datetime, format = "%y/%m/%d %H:%M:%S")
tt_1 = target.loc[target["Veh ID"]==117]
tt_2 = tt_1.loc[tt_1["Train Integrity"]=="Est"]
tt_3 = tt_2.drop(['Date',
 'Time',
 'VCC',
 'CHN',
 'Loop ID',
 'Veh ID',
 'Target Point',
 'Commanded Direction',
 'Door Cmd',
 'Max Vel',
 'Reply Type',
 'Active Passive',
 'Target Velocity',
 'Coded Command',
 'Non Safety Critical',
 'Last Response Valid',
 'Braking Curve',
 'Emergency Brake Control',
 'Average Gradient',
 'VCC.1',
 'CHN.1',
 'Response Type',
 'Operating Mode',
 'EB Status',
 'Door Status',
 'Train Integrity',
 'Active Passive Status',
 'RequestType1',
 'Actual Loop ID',
 'Actual Count Direction',
 'Veh Pos Number',
 
 'Type AB',
 'Veh ID.1',
 'Special Message',
 'Previous Loop ID',
 'Prev Entry Dir',
 'No Rx',
 
 
 'datetime'
 ], axis= 1)

tt_3 = tt_3.set_index("fixed")
del tt_1
del tt_2
#%% 
#now for the data
data["micro4"] = pd.to_datetime(data["X:Time (s)"], unit = "s")
offset = pd.to_datetime(head[3][23:40])
off2 = offset + pd.Timedelta(90, "s")
check = pd.Timedelta((off2.to_pydatetime() - data.micro4[0].to_pydatetime()))
data["dtime"] = data.micro4 + check

# data = data.drop(["X:Time (s)"], axis =1)
print(data.head())
# t1 = data["Y:g"]
t2 = data["Y:g"].values

#%%
data_fft = librosa.stft(t2[:], n_fft= 4096,  win_length = 2560)
del t2
#%%
# abs_data  = np.abs(data_fft)
hop = 2560 // 4
#%%

times = data.loc[data["X:Time (s)"]% (1/8) < 1/5120,:]
print(times)
print(times.shape)
#%%
#data_all =  librosa.stft(t2, n_fft= 4096,  win_length = 2560)
#%%
import matplotlib.pyplot as plt
import librosa.display as libd

fig, ax = plt.subplots()

img = libd.specshow(librosa.amplitude_to_db(np.abs(data_fft),

                                                       ref=np.max),

                               y_axis='linear', x_axis='time', sr=5120, ax=ax)

ax.set_title('Power spectrogram')

fig.colorbar(img, ax=ax, format="%+2.0f dB")
#%%
import pickle
#%%
# pickle.dump(data_fft, open("fft.p", "wb"))

# np.save("fft_np.npy", data_fft)
# pickle.dump(data_fft, open("fft.p", "wb"))

# np.save("abs_np.npy", abs_data)
#%%
# d2 = data.iloc[:,[1,3]]
# d2 = d2.set_index("dtime")
# mer = pd.merge(d2, tt_3, how = "outer", sort = True, left_index=True, right_index=True)
# mer = mer.drop_duplicates()
# #%%
# mer.loc["2019-03-05 10:31:11":"2019-03-05 13:16:44"]
# #%%
# mer = mer.fillna(method = "backfill")
# mer = mer.dropna()

#%%
delta = data.iloc[:,0].diff()
del data
#%%

seconds = np.arange(0, 11660*5120, 5120)
#%%
print(librosa.frames_to_samples(1382401, hop_length=2560//4, n_fft=4096))
#%%
tran = np.abs(data_fft.T)
del data_fft
new = np.zeros((tran.shape[0], 1))
rows = np.arange(0, 2049, 8)
for c,v in enumerate(rows):
    
    col = tran[:,v:min(v+8, 2048)]
    #print(" slice begining{} and end{}".format(v, min(v+8, 2048)))
    col = np.mean(col, axis=1)
    new = np.hstack((new, col.reshape(-1,1)))
new = new[:,1:-1]
print(new[:5])
del tran
del rows
del col

# #%%
# fig, ax = plt.subplots()

# img = libd.specshow(librosa.amplitude_to_db(new.T,

#                                                        ref=np.max),

#                                y_axis='linear', x_axis='time', sr=5120, ax=ax)

# ax.set_title('Power spectrogram')

# fig.colorbar(img, ax=ax, format="%+2.0f dB")
# #%%
# fig, ax = plt.subplots()

# img = libd.specshow(librosa.amplitude_to_db(np.abs(data_fft),

#                                                        ref=np.max),

#                                y_axis='linear', x_axis='time', sr=5120, ax=ax)

# ax.set_title('Power spectrogram')

# fig.colorbar(img, ax=ax, format="%+2.0f dB")
#%%
times = np.arange(0, new.shape[0]/8, 0.125).reshape(-1,1)
new = np.hstack((new, times))
print(new[:5])

#%%
d1 = pd.DataFrame(new)
d1.iloc[:,256] = pd.to_datetime(d1.iloc[:,-1], unit = "s", origin =off2)
d1.set_index(256, inplace=True)

#%%

mer = pd.merge(d1, tt_3, how = "left", sort = True, left_index=True, right_index=True)
mer = mer.drop_duplicates()
mer = mer.loc[str(tt_3.index[0]):str(tt_3.index[-1])]
mer = mer.fillna(method="backfill")
mer.to_csv("lstm_spectral_data.csv")
    
    
    
#%%
# rows = len(t2) //5120
# t2_ = t2[0:rows*5120]
# t3 = t2_.reshape(-1,5120)
# d2 = pd.DataFrame(t3)
# d2["time"] = np.arange(0,t3.shape[0])
# off2 = offset + pd.Timedelta(90, "s")
# d2["dtime"] = d2.time.apply( lambda x: pd.Timedelta(x, unit ="s") + off2)
# d2 = d2.set_index("dtime")
# mer = pd.merge(d2, tt_3, how = "outer", sort = True, left_index=True, right_index=True)
# mer = mer.drop_duplicates()
# mer = mer.dropna()
# mer.head()
