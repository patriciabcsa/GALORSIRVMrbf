#classifiers
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix ,classification_report 
from sklearn import linear_model
from sklearn.metrics import  r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle
from sklearn.model_selection import KFold

#load data
dfData = pd.read_csv('BD/BD_SEED/s10.csv')
dfData = dfData[dfData.label != 1] # este quita las etiqueta de los elementos neutrales

# Encode the classification labels to numbers
# Get classes and one hot encoded feature vectors
le = LabelEncoder()

le.fit(dfData['label'])
Y = le.transform(dfData['label'])
X = dfData.drop(['label','percol'], axis=1)

columnas = X.columns
new_set=[0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]

x=[x for x, j in zip(columnas, new_set) if j == 1]
X = pd.DataFrame(X, columns=x)
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


from sklearn.model_selection import cross_val_predict
model = linear_model.LinearRegression()
predicted = cross_val_predict(model, X, Y, cv=6)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(Y, predicted)
#ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

def k_fold(X,Y):
    cv = KFold(n_splits=3, random_state=0)

    for train_index, test_index in cv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test=k_fold(X,Y)
print(X_train)
    