
from csv import reader,writer
import statistics
import pandas as pd
import numpy 
import numpy as np
import csv
from sklearn import preprocessing
import pandas as pd


pd=pd.read_csv('BD/BD_SEED/s10.csv')
pd=pd['label']
arr=[]
limInferior=0.35
superior=0.7
for i in pd:
    
    if i<=limInferior:
        arr.append('0')
    if i>limInferior and i<superior:
        arr.append('1')
    if i>=superior:
        arr.append('2')
for i in arr:
    print(i)

