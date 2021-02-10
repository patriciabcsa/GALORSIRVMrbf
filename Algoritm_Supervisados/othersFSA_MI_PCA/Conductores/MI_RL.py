
# Feature Extraction with PCA
from sklearn.linear_model import LogisticRegression
import numpy
from pandas import read_csv
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.metrics import mean_absolute_error,accuracy_score,r2_score,confusion_matrix ,classification_report
from sklearn.metrics import confusion_matrix ,classification_report,r2_score 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
# load data
'''
dfData = pd.read_csv('BD/BD_SEED/BD_SEED.csv')
le = LabelEncoder()
le.fit(dfData['PRUEBA-NOON0-NIGHT1'])
Y = le.transform(dfData['PRUEBA-NOON0-NIGHT1'])
X = dfData.drop(['Sujeto','PRUEBA-NOON0-NIGHT1'], axis=1)
X=StandardScaler().fit_transform(X)

'''
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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42,shuffle=True)

# feature extraction
# el parametro k='all' cambia
fs = SelectKBest(score_func=mutual_info_classif, k=2)
fs.fit(X_train, y_train)
X_train = fs.transform(X_train)
X_test = fs.transform(X_test)
'''
variable=0.189
suma=0
contador=0
for i in range(len(fs.scores_)):
	#print('Feature %d: %f' % (i, fs.scores_[i]))
    if fs.scores_[i] > variable:
        contador+=1
        suma+=fs.scores_[i]
        print('Feature %d: %f' % (i, fs.scores_[i]))
        
print("resultado",suma/contador,contador)
''' 
'''
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.xlabel('Features')
plt.ylabel('Feature Importance Scores MI')
plt.title('Mutual information (MI)')
pyplot.show()
'''

#clasificador
regr = LogisticRegression(random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
Acuraccy_Train=regr.score(X_train, y_train)
print("\n")
print("     RL MI      ")
print("***** Train *****")
print("\n")
print("Accuracy Training    ",Acuraccy_Train)
#Imprimiendo Testing
print("***** Test *****")
print("Accuracy Testing     ",accuracy_score(y_test, y_pred))
'''
print("f1 score:                ",metrics.f1_score(y_test,y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
print("ROC:                     ",metrics.auc(fpr, tpr))
print("Specificity score ")
print("False Positive Rate)     ",1-(metrics.recall_score(y_test,y_pred, average='macro')))
print("Sensitivity (recall)     ",metrics.recall_score(y_test,y_pred, average='macro') )
print("Mean Absolute Error      ",metrics.mean_absolute_error(y_test,y_pred,))
print("Con_MatrixC")
print(cm)

print("classification_report")
print( metrics.classification_report(y_test,y_pred))



import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test,y_pred,normalize=True)
plt.title('Confusion Matrix RL MI')
plt.show()

from inspect import signature
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,average_precision_score
#Plotting the ROC curve 

fpr_logis, tpr_logis, thresholds_logis = roc_curve(y_test,y_pred)
fig, ax = plt.subplots(figsize = (10,7))
#plotting the "guessing" model
plt.plot([0, 1], [0, 1], 'k--')
#plotting the logistic model
plt.plot(fpr_logis, tpr_logis)
plt.fill_between(fpr_logis, tpr_logis, alpha=0.2, color='b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve RL MI: AUC={0:0.3f}'.format(roc_auc_score(y_test,y_pred)))
plt.show()


# Lift curve for the K-NN model

#KS plot for the K-NN model

target_proba = regr.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, target_proba)
plt.title('Lift curve RL MI')
plt.show()


skplt.metrics.plot_ks_statistic(y_test, target_proba)
plt.title('KS curve RL MI')
plt.show()
'''