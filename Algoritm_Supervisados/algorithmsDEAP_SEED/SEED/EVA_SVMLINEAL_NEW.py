#preprocessing
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy

#evaluation
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV
from sklearn.metrics import confusion_matrix ,classification_report,r2_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing,metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score,StratifiedKFold
from matplotlib import pyplot
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,KFold,cross_validate,learning_curve
import scikitplot as skplt

#load data
dfData = pd.read_csv('BD/BD_SEED/s10.csv')
dfData = dfData[dfData.label != 1] # este quita las etiqueta de los elementos neutrales

# Encode the classification labels to numbers
# Get classes and one hot encoded feature vectors
le = LabelEncoder()

le.fit(dfData['label'])
Y = le.transform(dfData['label'])
X = dfData.drop(['label','percol'], axis=1)

#columnas = X.columns
#new_set=[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]

#x=[x for x, j in zip(columnas, new_set) if j == 1]
#X = pd.DataFrame(X, columns=x)
#scaler = preprocessing.StandardScaler()
#X = scaler.fit_transform(X)


clf=svm.SVC(C=10,gamma=.001)

skfold = StratifiedKFold(n_splits=10,shuffle=True)
cv_scores = cross_validate(clf, X, Y, cv=skfold)
y_pred = cross_val_predict(clf, X, Y, cv=skfold)


#Imprimiendo Trainingm

#print(grid.best_estimator_)
print('\nTrain mean:              ',(cv_scores['train_score']).mean()*100.0)
print('Test mean:               ',cv_scores['test_score'].mean()*100.0,'+/-', cv_scores['test_score'].std() * 2)
print("Sensibilidad              ",metrics.recall_score(Y,y_pred, average='macro')*100.0)
fpr, tpr, thresholds = metrics.roc_curve(Y,y_pred)
#print("ROC:                     ",metrics.auc(fpr, tpr))
print("Precision                ",metrics.precision_score(Y,y_pred, average='macro') *100.0)

