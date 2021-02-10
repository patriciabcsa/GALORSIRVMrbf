
from csv import reader,writer
import statistics
import pandas as pd
import numpy 
import numpy as np
import csv
from sklearn import preprocessing
import pandas as pd
#dfData = pd.read_csv('BD/BD_conductores/BD_Bandas.csv')	#1subset
#dfData = pd.read_csv('BD/BD_conductores/alpha.csv')					#2subset
#dfData = pd.read_csv('BD/BD_conductores/Beta_Gamma.csv')				#3subset
#dfData = pd.read_csv('BD/BD_conductores/alpha_Beta.csv')				#4subset
#fData = pd.read_csv('BD/BD_conductores/alpha_Beta_Gamma.csv')		#5subset
#dfData = pd.read_csv('BD/BD_conductores/delta_alpha_beta.csv')		#6subset
#dfData = pd.read_csv('BD/BD_conductores/delta_alpha_gamma.csv')		#7subset
'''

arr=[]

with open('BD/BD_conductores/delta_alpha_gamma.csv', 'r') as read_obj:
    
    csv_reader = reader(read_obj)
    header = next(csv_reader) 
    if header != None:
        for row in csv_reader:
            new_set=[1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]
            x=[x for x, j in zip(row, new_set) if j == 1]
            scaler = preprocessing.StandardScaler()

            q=statistics.mean(list(map(float, x))) 
            arr.append(q)

numpy.savetxt("BD/BD_conductores/delta_alpha_gamma2.csv",arr,delimiter=",")
arr=np.array(arr)



pd.set_option('precision', 2)
#datos = dfData[:,:].flatten()

inf = arr.min()        # Limite inferior del primer intervalo
dif = arr.max()
sup = arr.max() + 1    # Limite superior del último intervalo

intervals = pd.interval_range(
                                start=inf,
                                end=sup,
                                periods=2,
                                name="Intervalo",
                                closed="left")

df = pd.DataFrame(index=intervals)
df["FreqAbs"] = pd.cut(arr, bins=df.index).value_counts()
df["Marca"]  = df.index.mid

df["LimInf"] = df.index.left
df["LimSup"] = df.index.right

print(df)

limInferior=[a for a in df["LimInf"]]
limSuper=[a for a in df["LimSup"]]

arr2=[]
cont=0

for n,i in enumerate(arr):
    if i<limInferior[1]:
        arr[n] = 1
    if i>=limInferior[1]:
        arr[n] = 2
    


a=numpy.savetxt("BD/BD_conductores/delta_alpha_gamma3.csv",arr,delimiter=",")
'''
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



#numpy.savetxt("BD/BD_conductores/alpha6.csv",arr,delimiter=",")

#my_filtered_csv = pd.read_csv(filename, usecols=['col1', 'col3', 'col7'])