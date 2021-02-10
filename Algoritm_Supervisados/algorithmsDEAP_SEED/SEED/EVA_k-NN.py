#Este programa esta listo para 
#classifiers
from sklearn.model_selection import GridSearchCV

#preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

#evaluation
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix ,classification_report ,roc_curve,roc_auc_score,precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import numpy
from sklearn.externals import joblib
import scikitplot as skplt
from inspect import signature
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn import preprocessing,metrics
from sklearn.model_selection import StratifiedKFold,KFold,cross_validate,learning_curve,cross_val_predict


#load data
dfData = pd.read_csv('BD/BD_SEED/s10.csv')
dfData = dfData[dfData.label != 1] # este quita las etiqueta de los elementos neutrales

# Encode the classification labels to numbers
# Get classes and one hot encoded feature vectors
le = LabelEncoder()

le.fit(dfData['label'])
Y = le.transform(dfData['label'])
X = dfData.drop(['label','percol'], axis=1)


clf = KNeighborsClassifier(n_neighbors=1) ##############################################################################################################
skfold = StratifiedKFold(n_splits=5,shuffle=False)
#grid = GridSearchCV(clf, cv=skfold, scoring='accuracy')
#grid.fit(X, Y)


cv_scores = cross_validate(clf, X, Y, cv=skfold,scoring='accuracy')
y_pred = cross_val_predict(clf, X, Y, cv=skfold)
print('\n')
#print(grid.best_params_)
#cross_val_score
print('\nTrain mean:              ',(cv_scores['train_score']).mean()*100.0)
print('Test mean:               ',cv_scores['test_score'].mean()*100.0,'+/-', cv_scores['test_score'].std() * 2)
print("Sensibilidad      ",metrics.recall_score(Y,y_pred, average='macro')*100.0)
fpr, tpr, thresholds = metrics.roc_curve(Y,y_pred)
#print("ROC:                     ",metrics.auc(fpr, tpr))
print("Precision:         ",metrics.precision_score(Y,y_pred, average='macro')*100.0 )
print('\n')
