import torch
import torch.nn as nn
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#now for the data
data["micro4"] = pd.to_datetime(data["X:Time (s)"], unit = "s")
offset = pd.to_datetime(head[3][23:40])
check = pd.Timedelta((offset.to_pydatetime() - data.micro4[0].to_pydatetime()))
data["dtime"] = data.micro4 + check

# data = data.drop(["X:Time (s)"], axis =1)
print(data.head())

#%%
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters = 2, random_state = 42, batch_size = 5120, max_iter = 5000).fit(data.iloc[:,0].values.reshape(-1,1))


#%%
data["lab"] = kmeans.labels_

#%%
tt_2["bin_vel"] = 0

tt_2.loc[tt_2["Actual Velocity"] > 0, "bin_vel"] = 1

#%%

tt_3 =  tt_2.drop(['Date', 'Time', 'VCC', 'CHN', 'Loop ID', 'Veh ID', 'Target Point',
       'Commanded Direction', 'Door Cmd', 'Max Vel', 'Reply Type',
       'Active Passive', 'Target Velocity', 'Coded Command',
       'Non Safety Critical', 'Last Response Valid', 'Braking Curve',
       'Emergency Brake Control', 'Average Gradient', 'VCC.1', 'CHN.1',
       'Response Type', 'Operating Mode', 'EB Status', 'Door Status',
       'Train Integrity', 'Active Passive Status', 'RequestType1',
       'Actual Loop ID', 'Actual Count Direction', 'Veh Pos Number',
        'Type AB', 'Veh ID.1', 'Special Message',
       'Previous Loop ID', 'Prev Entry Dir', 'No Rx',
        'datetime'], axis =1)
tt_3 = tt_3.drop_duplicates()

#%%

mer = pd.merge(data, tt_3, how="outer", left_on= "dtime", right_on="fixed")
mer = mer.sort_values("dtime")
