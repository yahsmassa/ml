import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import mglearn.plots as mplt
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler

def scalePlot():
    mplt.plot_scaling();plt.show()

def yscale(x,scaleType="s"):
    scaleType = scaleType.upper()
    y=None
    if x.ndim==1:
        x = x[:,None]
    
    if scaleType=="S":
        y= StandardScaler().fit_transform(x)
    elif scaleType=="M":
        y = MinMaxScaler().fit_transform(x)
    elif scaleType=="R":
        y = RobustScaler().fit_transform(x)

    return y

