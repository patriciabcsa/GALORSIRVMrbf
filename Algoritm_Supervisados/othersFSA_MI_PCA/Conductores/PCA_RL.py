
# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import mean_absolute_error,accuracy_score,r2_score,confusion_matrix ,classification_report
from sklearn.metrics import confusion_matrix ,classification_report,r2_score 
from sklearn import metrics
import matplotlib.pyplot as plt
# load data
#dfData = pd.read_csv('BD/BD_conductores/BD_Bandas.csv')	                #1subset
#dfData = pd.read_csv('BD/BD_conductores/alpha.csv')					#2subset
#dfData = pd.read_csv('BD/BD_conductores/Beta_Gamma.csv')				#3subset
#dfData = pd.read_csv('BD/BD_conductores/alpha_Beta.csv')				#4subset
#dfData = pd.read_csv('BD/BD_conductores/alpha_Beta_Gamma.csv')		    #5subset
#dfData = pd.read_csv('BD/BD_conductores/delta_alpha_beta.csv') 		#6subset
dfData = pd.read_csv('BD/BD_conductores/delta_alpha_gamma.csv')		#7subset
dfData = dfData.astype(np.float64)

le = LabelEncoder()
le_id = LabelEncoder()
le.fit(dfData['Tarea_R0_E1'])
Y = le.transform(dfData['Tarea_R0_E1'])
X = dfData.drop(['Suj','Tarea_R0_E1'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42,shuffle=True)

# feature extraction

pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance: %s" % explained_variance)

regr = LogisticRegression(random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
Acuraccy_Train=regr.score(X_train, y_train)
print("\n")
print("     RL PCA      ")
print("***** Train *****")
print("\n")
print("Accuracy Training    ",Acuraccy_Train)
#Imprimiendo Testing
print("***** Test *****")
print("Accuracy Testing     ",accuracy_score(y_test, y_pred))
