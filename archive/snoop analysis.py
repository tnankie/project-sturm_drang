# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:01:34 2021

@author: Grey Ghost
"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("./data/snoops.csv")
df["time"] = pd.to_datetime(df["Time"])
df2 = df.set_index("time")
df2["locc"] = df2["GW Area"].astype("category")
df3 = df2.between_time("10:35", "10:45")

#%%



#%%

df4 = df.iloc[400:2200, :]
df4 = df4.loc[df["Train Integrity"] == "Est"]

#%%



#%%
a = np.arange(600, 600 + df4.shape[0])


df4["x"] = a

#%%
df4.iloc[900:1500,:].plot(x="x", y= "Actual Velocity")

#%%
a =1 -.32
b = 5
c = a ** b
print(c)