#classifiers
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#evaluation
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import svm
from sklearn.metrics import  r2_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,accuracy_score,r2_score,confusion_matrix ,classification_report 
from sklearn import metrics
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import StratifiedKFold,KFold,cross_validate,learning_curve
import scikitplot as skplt
from sklearn.preprocessing import MinMaxScaler
from time import time

tiempo_inicial = time()
#dfData = pd.read_csv('BD/BD_DEAP/DEAP_valence.csv')
dfData = pd.read_csv('BD/BD_DEAP/pruebaDEAP.csv')
#dfData=dfData[dfData['user']==1]


le = LabelEncoder()
le.fit(dfData['valence'])
Y = le.transform(dfData['valence'])#etiqueta
X = dfData.drop(['user','video','op','valence','arousal'], axis=1)#datos
X = X.sample(n=120,axis='columns')
columnas = X.columns
new_set=[0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]

x=[x for x, j in zip(columnas, new_set) if j == 1]
X = pd.DataFrame(X, columns=x)
#scaler = MinMaxScaler()
#X = scaler.fit_transform(dfData)
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)


clf=svm.SVC()



skfold = StratifiedKFold(n_splits=5,shuffle=False)
cv_scores = cross_validate(clf, X, Y, cv=skfold,scoring='accuracy')
y_pred = cross_val_predict(clf, X, Y, cv=skfold)


#Imprimiendo Trainingm
tiempo_final = time() 
tiempo_ejecucion = tiempo_final - tiempo_inicial
#print(grid.best_estimator_)
print('\nTrain mean: \n             ',(cv_scores['train_score']).mean()*100.0)
print('Test mean:  \n             ',cv_scores['test_score'].mean()*100.0,'\n+/-', cv_scores['test_score'].std() * 2)
print("Sensibilidad\n      ",metrics.recall_score(Y,y_pred, average='macro')*100.0)
#fpr, tpr, thresholds = metrics.roc_curve(Y,y_pred)
print("Precision score:\n         ",metrics.precision_score(Y,y_pred, average='weighted') *100.0)
print('Timpo de ejecución',tiempo_ejecucion)

Y=Y.astype(np.uint8)
train_sizes, train_scores, test_scores = learning_curve(clf, X, Y, cv=skfold, train_sizes=np.linspace(.1, 1.0, 10), shuffle=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.figure()
plt.title("SVML Classifier S1")
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
plt.title('Confusion Matrix SVML S1')
#plt.savefig('BD/BD_conductores/knn_Confusion_Matrix_1.png')
plt.show()

	
'''
from sklearn.preprocessing import RobustScaler
from pandas import DataFrame
trans = RobustScaler()
X = trans.fit_transform(X)
X = DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 70% training and 30% test
from sklearn import svm

#Create a svm Classifier
Cs = np.logspace(-2, 10, 15)
gammas = np.logspace(-2, 10, 15)
param_grid = {'C': Cs, 'gamma' : gammas}
cv = StratifiedKFold(n_splits=10, shuffle=True)
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid,cv=cv)

#clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
grid_search.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = grid_search.predict(X_test)



from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print('\n')
print(grid_search.best_params_)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
'''

'''
# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''
