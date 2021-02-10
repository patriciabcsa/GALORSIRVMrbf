#classifiers
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#hPvE6GY9
#evaluation
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import numpy
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,accuracy_score,r2_score,confusion_matrix,classification_report 
from sklearn import metrics
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import StratifiedKFold,cross_validate,learning_curve
import scikitplot as skplt
from sklearn.preprocessing import MinMaxScaler
from time import time


tiempo_inicial = time() 

dfData = pd.read_csv('BD/BD_DEAP/pruebaDEAP.csv')


le = LabelEncoder()
le.fit(dfData['valence'])
Y = le.transform(dfData['valence'])#etiqueta

X = dfData.drop(['user','video','op','valence','arousal'], axis=1)#datos
X = X.sample(n=40,axis='columns')
columnas = X.columns
new_set=[0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0]
x=[x for x, j in zip(columnas, new_set) if j == 1]

X = pd.DataFrame(X, columns=x)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)



Cs=np.logspace(-2, 10, 15)#np.logspace(-2, 10, 15)#Cs = [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]  #Cs = [0.001, 0.01, 0.1, 1, 10]
gammas=[0.00001,0.0001,0.001, 0.01, 0.1,1, 10]#gammas = [0.00001,0.0001,0.001, 0.01, 0.1,1, 10] #gammas = [0.001, 0.01, 0.1, 1]
kernels=['rbf']

param_grid = {'C': Cs, 'gamma' : gammas,'kernel':kernels}
param_grid=dict(C=Cs,gamma=gammas,kernel=kernels)

# creando modelo
skfold = StratifiedKFold(n_splits=5,shuffle=True)
svc = svm.SVC()
svclassifier= GridSearchCV(svc, param_grid, scoring='accuracy')#,refit = True, verbose = 3)

svclassifier= svm.SVC(C= 10, gamma= 2, kernel='rbf') #############################################################################################################
#svclassifier.fit(X, Y)#{'C': 0.1, 'gamma': 0.0001, 'kernel': 'rbf'


cv_scores = cross_validate(svclassifier, X, Y, cv=skfold)
y_pred = cross_val_predict(svclassifier, X, Y, cv=skfold)

print(len(y_pred))

tiempo_final = time() 
tiempo_ejecucion = tiempo_final - tiempo_inicial
print("\n")
#print(svclassifier.best_params_)
#cross_val_score
print(X)
print(cv_scores)

print('\nTrain mean:              ',(cv_scores['train_score']).mean()*100.0)
print('Test mean:               ',cv_scores['test_score'].mean()*100.0,'\n+/-', cv_scores['test_score'].std() * 2)
print("Sensitivity:             ",metrics.recall_score(Y,y_pred, average='macro')*100.0)
fpr, tpr, thresholds = metrics.roc_curve(Y,y_pred)
#print("ROC:                     ",metrics.auc(fpr, tpr))
print("Precision:               ",metrics.precision_score(Y,y_pred, average='macro') *100.0)
print('Timpo de ejecución',tiempo_ejecucion)
for i in range(len(X)):
	print("X=%s, Predicted=%s" % (Y[i], y_pred[i]))

Y=Y.astype(np.uint8)
train_sizes, train_scores, test_scores = learning_curve(svclassifier, X, Y, cv=skfold, train_sizes=np.linspace(.1, 1.0, 10), shuffle=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.figure()
plt.title("SVMRBF Classifier S1")
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
plt.title('Confusion Matrix SVMRBF S1')
#plt.savefig('BD/BD_conductores/knn_Confusion_Matrix_1.png')
plt.show()
