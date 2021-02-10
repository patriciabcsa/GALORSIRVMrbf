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

#load data
dfData = pd.read_csv('data3.csv',delimiter=",")
dfData = dfData.astype(np.float64)

le = LabelEncoder()
le.fit(dfData['Etiqueta'])
Y = le.transform(dfData['Etiqueta'])
X = dfData.drop(['Data_Id','Etiqueta','Sesiones '], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)




def svc_param_selection_rbf(X, y, nfolds,X_t,y_t):
    Cs = [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]  #Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [ 0.00001,0.0001,0.001, 0.01, 0.1] #gammas = [0.001, 0.01, 0.1, 1]
    kernels=['rbf']
    param_grid = {'C': Cs, 'gamma' : gammas,'kernel':kernels}
    clf = GridSearchCV(svm.SVC(kernels), param_grid, cv=nfolds)
  
    clf.fit(X, y)
   
    predict=clf.predict(X_t)

    results = confusion_matrix(y_t, predict) 
    scr=clf.score(X, y)
    clasificacion=classification_report(y_t, predict)
    lin_svm_test = clf.score(X_t, y_t)
    print (clf.best_params_)
    return scr,results,clasificacion,lin_svm_test

def svc_param_selection_lineal(X, y, nfolds,X_t,y_t):
    #Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100,10e5]  
    #kernels=['linear']
    #gammas = [0.1,0.01]
    #param_grid = {'C': Cs,'kernel':kernels, 'gamma':gammas}
    #clf = GridSearchCV(svm.SVC(kernels), param_grid, cv=nfolds)
    #clf = SVC(gamma=5)
    clf = SVC(gamma='auto')
    clf.fit(X, y)
 
    predict=clf.predict(X_t)
    results = confusion_matrix(y_t, predict) 
    scr=clf.score(X, y)
    clasificacion=classification_report(y_t, predict)
    lin_svm_test = clf.score(X_t, y_t)
    return scr,results,clasificacion,lin_svm_test

def knn_param_selection(X,y,nfolds,X_t,y_t):

    #k_range = list(range(1,3))
    """
    #weight_options = ["uniform", "distance"]
    #metric=['euclidean','manhattan']
    #param_grid = dict(n_neighbors = k_range, weights = weight_options, metrics=metric)
    """
    #param_grid = dict(n_neighbors = k_range)
    #knn = KNeighborsClassifier()
    #clf = GridSearchCV(knn, param_grid)
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(X,y)
    predict=clf.predict(X_t)
    
    results = confusion_matrix(y_t, predict) 
    #clf.best_params_
    #clf.best_estimator_
    #clf.best_score_,
    scr=clf.score(X, y)
    clasificacion=classification_report(y_t, predict)
    knn_test = clf.score(X_t, y_t)
    return scr,results,clasificacion,knn_test



def RL_param_selection(X,y,X_t,y_t):
    
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(X)
    x_test = sc_x.transform(X_t)
    
    penalty = ['l1', 'l2']
    C = [0.001, 0.01, 0.10, 0.1,1, 10, 25, 50, 100, 1000]
    hyperparameters = dict(C=C, penalty=penalty)
   
    logitg = LogisticRegression()
    logit = GridSearchCV(logitg, hyperparameters, cv=10, verbose=0)
    logit = LogisticRegression()
    logit.fit(x_train, y)

    y_predicted = logit.predict(x_test)
    results = confusion_matrix(y_t, y_predicted)

    clasificacion=classification_report(y_t, y_predicted)
    RL_test = logit.score(X_t, y_t)
   
  
    return results,clasificacion,RL_test


nfolds=10
SVC_rbf,SVC_rbf_cm,clasificacionrbf,lin_svm_test=svc_param_selection_rbf(X_train, y_train,nfolds,X_test,y_test)
SVC_lineal,SVC_lineal_cm,clasificacionl,lin_svm_testlineal=svc_param_selection_lineal(X_train, y_train,nfolds,X_test,y_test)
k_NN,k_NN_cm,clasificacionknn,knn_test=knn_param_selection(X_train, y_train,nfolds,X_test,y_test)
RL_cm,clasificacionrl,RL_test=RL_param_selection(X_train, y_train,X_test,y_test)
lin_svm_test=lin_svm_test*100
print('\n')
#print ("SVC_rbf         ", SVC_rbf)
print ("SVC_rbf TEST    ",lin_svm_test)
#print ("clasificacionrbf ",clasificacionrbf)
#print ("SVC_rbf_par     ",SVC_rbf_par)
print ("SVC_rbf_cm")
print(SVC_rbf_cm)
print()
lin_svm_testlineal=lin_svm_testlineal*100
#print ("SVC_lineal      ",SVC_lineal)
print ("SVC_lineal TEST ",lin_svm_testlineal)
#print ("clasificacionl  ",clasificacionl)
print ("SVC_lineal_cm   ")
print(SVC_lineal_cm)
print()
knn_test=knn_test*100
#print ("k_NN            ",k_NN)
print ("k_NN TEST        ",knn_test)
#print ("clasificacionknn",clasificacionknn)
#print ("k_NN_par        ",k_NN_par)
print ("k_NN_cm         ")
print(k_NN_cm)
print()
RL_test=RL_test*100
#print ("RL              ",RL)
print ("RL TEST         ",RL_test)
#print ("clasificacionrl ",clasificacionrl)
print ("RL_cm           ")
print(RL_cm)
print('\n')





plt.clf()
fig = plt.figure()
plt.suptitle('Confusion Matrix',  fontsize=12)

plt.subplot(2, 2, 1)
plt.imshow(SVC_rbf_cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Relajado','Estresado']
plt.title('SVM RBF \nAccuracy:{0:.2f}'.format(lin_svm_test),fontsize=10)
plt.ylabel('True label',fontsize=8)
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=0,fontsize=8)
plt.yticks(tick_marks,classNames,fontsize=8)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+"="+str(SVC_rbf_cm[i][j]),fontsize=10)



plt.subplot(2, 2, 2)
plt.imshow(SVC_lineal_cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Relajado','Estresado']
plt.title('SVM lineal \nAccuracy:{0:10.2f}'.format(lin_svm_testlineal),fontsize=10)
plt.xticks(tick_marks, classNames, rotation=0,fontsize=8)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+"="+str(SVC_lineal_cm[i][j]),fontsize=10)

plt.subplot(2, 2, 3)
plt.imshow(k_NN_cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Relajado','Estresado']
plt.title('k-NN\nAccuracy:{0:10.2f}'.format(knn_test),fontsize=10)
plt.ylabel('True label',fontsize=8)
plt.xlabel('Predicted label',fontsize=8)
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=0,fontsize=8)
plt.yticks(tick_marks, classNames,fontsize=8)
s = [['TP','FP'], ['FN', 'TN']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+"="+str(k_NN_cm[i][j]),fontsize=10)

plt.subplot(2, 2, 4)
plt.imshow(RL_cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Relajado','Estresado']
plt.title('RL \nAccuracy:{0:10.2f}'.format(RL_test),fontsize=10)
plt.xlabel('Predicted label',fontsize=8)
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=0,fontsize=8)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+"="+str(RL_cm[i][j]),fontsize=10)
plt.show()