import csv
import random
import numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from scipy import interpolate
from scoop import futures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from time import time

# Read in data from CSV

dfData = pd.read_csv('BD/BD_conductores/BD_Bandas.csv')



le = LabelEncoder()

le.fit(dfData['Tarea_R0_E1'])
allClasses = le.transform(dfData['Tarea_R0_E1'])
allFeatures = dfData.drop(['Tarea_R0_E1','Suj'], axis=1)






######## GLOBAL VARIABLES OF GA ########

# Create Individual 
## The individual is defined
## The first individual created will be a simple list
## To create an Individual class is using the creator, that will inherit from the standard list type and have a fitness parameters.
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # The configuration of the genes of the chromosome is defined
creator.create("Individual", list, fitness=creator.FitnessMax) # Create chromosomes

# Create Toolbox Population...
# Creates 3 aliases in the toolbox; attr_float, individual and population
toolbox = base.Toolbox() #contains the evolutionary operators
toolbox.register("attr_bool", random.randint, 0, 1) 
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(allFeatures.columns)) # Define Indiv.
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Create population







######## FF  ########

# Form training, test, and validation sets
X_trainAndTest, X_validation, y_trainAndTest, y_validation = train_test_split(allFeatures, allClasses, test_size=0.20,shuffle=True) #validation
X_train, X_test, y_train, y_test = train_test_split(X_trainAndTest, y_trainAndTest, test_size=0.20,shuffle=True) #training, test

# Feature subset fitness function
def getFitness(individual, X_train, X_test, y_train, y_test):

	chro = [index for index in range(len(individual)) if individual[index] == 0]
	
    # Ensure that the array isnt empty
	X_trainnotempty = X_train.drop(X_train.columns[chro], axis=1)
	X_testnotempty = X_test.drop(X_test.columns[chro], axis=1)
	    
    # Apply logistic regression on the data and calculate fit
	C_param_range = [0.00001,0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,10000]
	for i in C_param_range:
		clf = LogisticRegression( C = i,random_state=0)
		clf.fit(X_trainnotempty, y_train)
		predictions = clf.predict(X_testnotempty)
		accuracy = accuracy_score(y_test, predictions)

		pvalue=[(pvalue(clf, X_trainnotempty))]

	# Return calculated accuracy as fitness
	return (accuracy,pvalue)







######## EVOLUTIONARY PROCESS	########
# Continue filling toolbox...toolbox contains
toolbox.register("evaluate", getFitness, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
toolbox.register("mate", tools.cxTwoPoint) 
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)







######## STORED A INDIVIDUAL BASED ON FF PARAMETERS		########

# the first element of the getHof is the individual that has the best first fitness parameters value
# according to the obtained parameters from the FF.
def getHof():
	# Initialize variables to evolutor process
	# Defines the size of the population
	numPop =100#100#100#50 #30 #100
	numGen =30#30#80#100#10
	pop = toolbox.population(n=numPop) #toolbox population contains
	
	logbook = tools.Logbook() #Stores the parameters information of each generation
	hof = tools.HallOfFame(numPop * numGen) # contains the best individual in the population during the evolution.
	
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)
	
	# Launch genetic algorithm
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.66, mutpb=0.8, ngen=numGen, stats=stats, halloffame=hof, verbose=True)#config.

	return hof,log





######## ORGANIZE INDIVIDUALS BASED ON PARAMETERS	######## 
def getMetrics(hof):
	# Get list of percentiles in the hall of fame
	percList = [i / (len(hof) - 1) for i in range(len(hof))]
	
	# Gather fitness data from each percentile
	testAccList = []
	validationAccList = []
	individualList = []
	
	for individual in hof:
		testAcc = individual.fitness.values
		validationAcc = getFitness(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)
		testAccList.append(testAcc[0])
		validationAccList.append(validationAcc[0])
		individualList.append(individual)
		
	testAccList.reverse()
	validationAccList.reverse()
	
	return testAccList, validationAccList, individualList, percList
 


if __name__ == '__main__':

	# First, the chromosome is created and sent to the fittnes function to evaluate it.
	individual = [1 for i in range(len(allFeatures.columns))]
	testAccuracy = getFitness(individual, X_train, X_test, y_train, y_test)
	validationAccuracy= getFitness(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)
	print('\nTest accuracy with all features: \t' + str(testAccuracy[0]))
	print('Validation accuracy with all features: \t' + str(validationAccuracy[0]) + '\n')
	
	# Now, the genetic algorithm is applied to choose a subset of characteristics
	hof,log = getHof()
	testAccuracyList, validationAccList, individualList, percList = getMetrics(hof)
	
	# list of subsets 
	maxValAccIndicies = ([index for index in range(len(validationAccList)) if validationAccList[index] == max(validationAccList)])
	maxIndividuals =([individualList[index] for index in maxValAccIndicies])
	maxSubsets = ([[list(allFeatures)[index] for index in range(len(individual)) if individual[index] == 1] for individual in maxIndividuals])

	
	for index in range(len(maxValAccIndicies)):
		if min(maxSubsets):
			Individual=((maxIndividuals[index]))
			print(Individual)





######## SVMRBF	######## 
# NEW DATA STRUCTURE
	
	x=[x for x, j in zip(names, list(Individual)) if j == 1]
	print(x)
	X = pd.DataFrame(allFeatures, columns=x)

	scaler = MinMaxScaler()
	#scaler = preprocessing.StandardScaler()
	X = scaler.fit_transform(X)


	###### CREATE THE CLAFCIFIER
	Cs= [ 0.1, 1, 10, 100, 1000]#np.logspace(-2, 10, 15)
	gammas=[ 0.0001, 0.001, 0.01, 0.1, 1] 
	kernels=['rbf']

	param_grid=dict(C=Cs,gamma=gammas,kernel=kernels)

	# CV K-10
	skfold = StratifiedKFold(n_splits=5,shuffle=True)

	svc = svm.SVC()
	svclassifier= GridSearchCV(svc, param_grid, scoring='accuracy')#,refit = True, verbose = 3)
	
	svclassifier.fit(X, allClasses)
	#scores = cross_val_score(svclassifier, X, allClasses, cv=skfold) #TRAINING 
	#predictions = cross_val_predict(svclassifier, X, allClasses, cv=skfold) #PREDICT

	cv_scores = cross_validate(svclassifier, X, allClasses, cv=skfold) #TRAINING 
	y_pred = cross_val_predict(svclassifier, X, allClasses, cv=skfold)  #PREDICT

	##############################


	tiempo_final = time() 
	tiempo_ejecucion = tiempo_final - tiempo_inicial


    
	print('\n')
	print('\n---Optimal Feature Subset(s)---')
	Validation=str(validationAccList[maxValAccIndicies[index]])
	Individual=((maxIndividuals[index]))
	Numbero=str(len(maxSubsets[index]))
	Feature=str(maxSubsets[index])
	
	print('Validation Fit: \t\t' + Validation)
	print('Individual')
	print(Individual)
	print('Number Features In Subset: \t' +Numbero)
	print('Feature Subset: ' + Feature)
	print(log)
	print('\n')	
	
	######## STORE FILE ######## 
	fichero = open('BD/DEAP_valence_result.txt','a') 

	fichero.write('\n')
	fichero.write(str(['ACCURACY:	',Validation]))
	fichero.write('\n')
	fichero.write(str(['NUMBER:		',Numbero]))
	fichero.write('\n')
	fichero.write(str(['INDIVIDUAL:	',Individual]))
	fichero.write('\n')
	fichero.write(str(['FEATURE:		',Feature]))
	fichero.close()
	

	#Now, we plot the test and validation classification accuracy to see how these numbers change as we move from our worst feature subsets to the 
	#best feature subsets found by the genetic algorithm.
	
	# Calculate best fit line for validation classification accuracy (non-linear)
	tck = interpolate.splrep(percList, validationAccList, s=5.0)
	ynew = interpolate.splev(percList, tck)
	
	
	f = plt.figure(1)
	gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
	plt.plot(gen, avg, label="Best Fitness",color='b')
	plt.plot(gen, max_, label="Maximum Fitness",marker='o',color='g')
	plt.plot(gen, min_, label="Minimum Fitness")
	plt.xlabel("Number of Generations")
	plt.ylabel("Fitness Values")
	plt.legend(loc="lower right")
	plt.title(' Fitness')
	f.show()
	#plt.savefig('Subset_2_Fitness_alpha.png')

	e = plt.figure(2)
	plt.plot(percList, validationAccList, marker='o', color='g')
	plt.plot(percList, ynew, color='b')
	plt.title('Validation Set Classification Accuracy ')
	plt.xlabel('# Population ')
	plt.ylabel(' Accuracy')
	plt.legend(loc="lower right")
	#plt.savefig('Subset_2_alpha.png')
	e.show()

	print("\n")
	print('\nTrain mean:              ',(cv_scores['train_score']).mean()*100.0)
	print('Test mean:               ',cv_scores['test_score'].mean()*100.0,'\n+/-', cv_scores['test_score'].std() * 2)
	print("Sensitivity:             ",metrics.recall_score(allClasses,y_pred, average='macro')*100.0)
	fpr, tpr, thresholds = metrics.roc_curve(allClasses,y_pred)
	#print("ROC:                     ",metrics.auc(fpr, tpr))
	print("Precision:               ",metrics.precision_score(allClasses,y_pred, average='macro') *100.0)
	print('Timpo de ejecución',tiempo_ejecucion)
	print(svclassifier.best_params_)
	input()
