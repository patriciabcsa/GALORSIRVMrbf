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

#columnas = X.columns
#new_set=[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]

#x=[x for x, j in zip(columnas, new_set) if j == 1]
#X = pd.DataFrame(X, columns=x)
##scaler = preprocessing.StandardScaler()
#X = scaler.fit_transform(X)



#k_range = list(range(10, 20))
#param_grid = dict(n_neighbors=k_range)
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
"""
train_sizes, train_scores, test_scores = learning_curve(clf, X, Y, cv=skfold, train_sizes=np.linspace(.1, 1.0, 10), shuffle=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.figure()
plt.title("k-NN Classifier S10")
plt.legend(loc="best")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.gca().invert_yaxis()
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")   
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.ylim(-.1,1.1)
#plt.savefig('BD/BD_conductores/Knn_learning_curve_1.png')
plt.show()

skplt.metrics.plot_confusion_matrix(Y,y_pred,normalize=True)
plt.title('Confusion Matrix k-NN S10')
#plt.savefig('BD/BD_conductores/knn_Confusion_Matrix_1.png')
plt.show()
"""