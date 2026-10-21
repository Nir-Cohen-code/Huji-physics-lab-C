# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:32:03 2026

@author: cohen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


p="..\\week1 day 1 samples\\"
df1 = pd.read_csv(p+"iv room temp.csv",header=None, usecols=[3, 4, 9, 10])
df2 = pd.read_csv(p+"iv cooled xy flipped.csv",header=None, usecols=[3, 4, 9, 10])
df3 = pd.read_csv(p+"iv cooled more xy flipped.csv",header=None, usecols=[3, 4, 9, 10])

window=10
#bla 
I_hot=df1[4].rolling(window=window).mean()
V_hot=df1[10].rolling(window=window).mean()

I_cold=df3[10].rolling(window=window).mean()
V_cold=df3[4].rolling(window=window).mean()

plt.plot(I_hot,V_hot,'.')
plt.plot(I_cold,V_cold,'.')


