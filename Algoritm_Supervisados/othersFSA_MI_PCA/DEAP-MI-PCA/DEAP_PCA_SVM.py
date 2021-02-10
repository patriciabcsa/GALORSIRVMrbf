
# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import mean_absolute_error,accuracy_score,r2_score,confusion_matrix ,classification_report
from sklearn.metrics import confusion_matrix ,classification_report,r2_score 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
# load data

dfData = pd.read_csv('BD/BD_SEED.csv')
le = LabelEncoder()
le.fit(dfData['PRUEBA-NOON0-NIGHT1'])
Y = le.transform(dfData['PRUEBA-NOON0-NIGHT1'])
X = dfData.drop(['Sujeto','PRUEBA-NOON0-NIGHT1'], axis=1)
X=StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,shuffle=True)

# feature extraction

pca = PCA(n_components=9)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


#clasificador
clf = make_pipeline(StandardScaler(), SVC(kernel='linear',probability=True))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
Acuraccy_Train=clf.score(X_train, y_train)


print("\n")
print("     SVM PCA      ")
print("***** Train *****")
print("\n")
print("Accuracy Training    ",Acuraccy_Train)
#Imprimiendo Testing
print("***** Test *****")
print("Accuracy Testing     ",accuracy_score(y_test, y_pred))
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
plt.title('Confusion Matrix SVM PCA')
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
plt.title('ROC curve SVM PCA: AUC={0:0.3f}'.format(roc_auc_score(y_test,y_pred)))
plt.show()


# Lift curve for the K-NN model

#KS plot for the K-NN model

target_proba = clf.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, target_proba)
plt.title('Lift curve SVM PCA')
plt.show()


skplt.metrics.plot_ks_statistic(y_test, target_proba)
plt.title('KS curve SVM PCA')
plt.show()