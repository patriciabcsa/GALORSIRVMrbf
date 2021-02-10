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

'''
train_sizes, train_scores, test_scores = learning_curve(clf, X, Y, cv=skfold, train_sizes=np.linspace(.1, 1.0, 10), shuffle=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.figure()
plt.title("SVMLineal Classifier S10")
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
plt.title('Confusion Matrix SVMLineal S1')
#plt.savefig('BD/BD_conductores/knn_Confusion_Matrix_1.png')
plt.show()
'''