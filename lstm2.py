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
# then filtering the dataset
target["datetime"] = target.apply(lambda x: str(x.Date) + " " + str(x.Time), axis=1)
target["fixed"] = pd.to_datetime(target.datetime, format="%y/%m/%d %H:%M:%S")
tt_1 = target.loc[target["Veh ID"] == 117]
tt_2 = tt_1.loc[tt_1["Train Integrity"] == "Est"]
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
                  ], axis=1)
tt_3 = tt_3.set_index("fixed")
# now for the data
data["micro4"] = pd.to_datetime(data["X:Time (s)"], unit="s")
offset = pd.to_datetime(head[3][23:40])
check = pd.Timedelta((offset.to_pydatetime() - data.micro4[0].to_pydatetime()))
data["dtime"] = data.micro4 + check

# data = data.drop(["X:Time (s)"], axis =1)
print(data.head())
t1 = data["Y:g"]
t2 = data["Y:g"].values
rows = len(t2) // 5120
t2_ = t2[0:rows * 5120]
t3 = t2_.reshape(-1, 5120)
d2 = pd.DataFrame(t3)
d2["time"] = np.arange(0, t3.shape[0])
off2 = offset + pd.Timedelta(90, "s")
d2["dtime"] = d2.time.apply(lambda x: pd.Timedelta(x, unit="s") + off2)
d2 = d2.set_index("dtime")
mer = pd.merge(d2, tt_3, how="outer", sort=True, left_index=True, right_index=True)
mer = mer.drop_duplicates()
mer = mer.dropna()
mer.head()
