from numpy import *
from matplotlib import pyplot as plt
from sklearn.metrics import mutual_info_score

import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
########################################### Load Data From CVS

dfData = pd.read_csv('BD/BD_conductores/BD_Bandas.csv')
dfData = dfData.astype(np.float64)


scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(dfData)
print(X)
np.savetxt("BD/binaria_BD2.csv",
           X,
           delimiter=",")