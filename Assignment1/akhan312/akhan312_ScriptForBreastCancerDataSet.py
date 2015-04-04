from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.externals.six import StringIO 
import pydot
from sklearn import cross_validation
import pybrain #install using pip install -i https://pypi.binstar.org/pypi/simple pybrain
from pybrain.datasets import ClassificationDataSet
import copy
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
import time

#*******************All Functions*****************
print (time.localtime())
# this function takes a input of dict of data with as InputData, keys to values that need to converted from string to float, and key for the classification field
# The function returns InputData in the same dict form but number values changed from string to float and with classification data removed.
# The function also returns all the classification values in the list called ExtractedTarget
def extract_target_ConvertStringNumber(InputData, numStringKeys, classKey):
	for d in InputData:
		ExtractedTarget.append(d[classKey])
		del d[classKey]
		for k in numStringKeys:
			d[k] = float(d[k])

	return (InputData, ExtractedTarget)


def KFolds_CrossVal(data, target, learners, folds):
    """
    Compute a k-folds validation on a scikit-learn data set for several models
    """
    learner_scores = []
    for model in learners:
        scores = cross_validation.cross_val_score(model, data, target, cv=folds)
        learner_scores.append(sum(scores) / len(scores))
    return learner_scores

def plot_validation_curve(estimator, X, y, param_name, param_range, addition_graph_points, graph_title, graph_xlabel, graph_ylabel, ylim, cv=5, scoring="accuracy"):
	
	cv_train_scores, cv_test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring)

	cv_train_scores_mean = np.mean(cv_train_scores, axis=1)
	cv_train_scores_std = np.std(cv_train_scores, axis=1)
	cv_test_scores_mean = np.mean(cv_test_scores, axis=1)
	cv_test_scores_std = np.std(cv_test_scores, axis=1)

	plt.title(graph_title)
	plt.xlabel(graph_xlabel)
	plt.ylabel(graph_ylabel)
	plt.ylim(*ylim)

	plt.fill_between(param_range, cv_train_scores_mean - cv_train_scores_std, cv_train_scores_mean + cv_train_scores_std, alpha=0.1, color="r")
	plt.fill_between(param_range, cv_test_scores_mean - cv_test_scores_std,cv_test_scores_mean + cv_test_scores_std, alpha=0.1, color="b")
	plt.plot(param_range, cv_train_scores_mean, 'o-', color="r", label="Cross Validation Training score")
	plt.plot(param_range, cv_test_scores_mean, 'o-', color="b",label="Cross Validation Test Score")
	
	for gp in addition_graph_points:
		plt.plot(param_range, gp['data'], 'o-', color=gp['color'],label=gp['label'])

	plt.legend(loc="best")
	plt.savefig('plots/BreastCancerPlots/'+graph_title+'.png')
	plt.close()
	#plt.show()

def plot_learning_curve(estimator, X, y, graph_title, graph_xlabel, graph_ylabel, ylim, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(graph_title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(graph_xlabel)
    plt.ylabel(graph_ylabel)

    train_sizes, cv_train_scores, cv_test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    cv_train_scores_mean = np.mean(cv_train_scores, axis=1)
    cv_train_scores_std = np.std(cv_train_scores, axis=1)
    cv_test_scores_mean = np.mean(cv_test_scores, axis=1)
    cv_test_scores_std = np.std(cv_test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, cv_train_scores_mean - cv_train_scores_std,
                     cv_train_scores_mean + cv_train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, cv_test_scores_mean - cv_test_scores_std,
                     cv_test_scores_mean + cv_test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, cv_train_scores_mean, 'o-', color="r",
             label="Cross Validation Training score")
    plt.plot(train_sizes, cv_test_scores_mean, 'o-', color="g",
             label="Cross Validation Test Score")

    plt.legend(loc="best")
    plt.savefig('plots/BreastCancerPlots/'+graph_title+'.png')
    plt.close()
    #plt.show()

#*******************End of Functions**************
#*******************Prepare Data & Target for Estimators******************
print("Machine Learning Program Started")

InputData = list(csv.DictReader(open('BreastCancerWisconsinDataset.txt', 'rU')))

numStringKeys = ['ClumpThickness','UniformityCellSize','UniformityCellShape','MarginalAdhesion','SingleEpithelialCellSize','BareNuclei','BlandChromatin','NormalNucleoli','Mitoses']
classKey = 'Class'
ExtractedTarget = []

FormatedRawData, ExtractedTarget = extract_target_ConvertStringNumber(InputData, numStringKeys, classKey)

ExtractedTarget_array = np.array(ExtractedTarget)

le = preprocessing.LabelEncoder()
le.fit(ExtractedTarget_array)

Target = le.transform(ExtractedTarget_array)

vec = DictVectorizer()
Data = vec.fit_transform(FormatedRawData).toarray()

# Normalize data
Data = preprocessing.normalize(Data, norm='l2', axis=0)

# split data into training and test sets
DataTrain, DataTest, TargetTrain, TargetTest = train_test_split(Data, Target, test_size=0.33, random_state=0)

print("\n")
print("Th complete dataset shape is : ", Data.shape)
print("Th complete target shape is : ", Target.shape)
print("The training data shape is (2/3 of complete dataset): ", DataTrain.shape)
print("The training target shape is (2/3 of complete target): ", TargetTrain.shape)
print("The test data shape is (1/3 of complete dataset): ", DataTest.shape)
print("The test target shape is (1/3 of complete target): ", TargetTest.shape)
print("\n")

#prepare data for pybrain
number_of_columns = Data.shape[1]
PyBData = ClassificationDataSet(number_of_columns, 1, nb_classes=2)
PyBDataTrain = ClassificationDataSet(number_of_columns, 1, nb_classes=2)
PyBDataTest = ClassificationDataSet(number_of_columns, 1, nb_classes=2)

for i in xrange(len(Data)):
	PyBData.appendLinked(Data[i], Target[i])
	
for i in xrange(len(DataTrain)):
	PyBDataTrain.appendLinked(DataTrain[i], TargetTrain[i])
	
for i in xrange(len(DataTest)):
	PyBDataTest.appendLinked(DataTest[i], TargetTest[i])

#*******************End of Preparing Data & Target for Estimators******************

#*******************Decision Tree Classification******************
print("Entering Decision Tree Classifier with starting time", time.localtime())

clf_dt = tree.DecisionTreeClassifier(criterion="entropy")
clf_dt = clf_dt.fit(DataTrain, TargetTrain)

print("Completed initial fit", time.localtime())

print ("Training accuracy of Decision Tree with default settings and criterion=entropy", clf_dt.score(DataTrain, TargetTrain))
print ("Testing accuracy of Decision Tree with default settings and criterion=entropy", clf_dt.score(DataTest, TargetTest))

# create plot of decision tree learning curve
graph_title = "Decision Tree Learning Curve"
graph_xlabel = "Number of Samples"
graph_ylabel = "Score"
ylim = (.7, 1.1)
estimator = tree.DecisionTreeClassifier(criterion="entropy")
plot_learning_curve(estimator, DataTrain, TargetTrain, graph_title, graph_xlabel, graph_ylabel, ylim)

# create points for validation curve plotting of the accuracy score of full Training and Test data on varying values of min_sample_split
min_samples_split = np.arange(5, 130, 5)
full_train_score = []
unseen_test_scores = []

for m in min_samples_split:
	clf_dt = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=m)
	clf_dt = clf_dt.fit(DataTrain, TargetTrain)
	full_train_score.append(clf_dt.score(DataTrain, TargetTrain))
	unseen_test_scores.append(clf_dt.score(DataTest, TargetTest))

# Variable for plotting validation curve
addition_graph_points = [{'data':full_train_score, 'color':'y', 'label': 'Full Training Data Score'}, 
	{'data':unseen_test_scores, 'color':'g', 'label': 'Full Test Data Score'}]
graph_title = "Decision Tree Validation Curve"
graph_xlabel = "Min Samples Split"
graph_ylabel = "Score"
ylim = (.7, 1.1)
param_name = "min_samples_split"
estimator = tree.DecisionTreeClassifier(criterion="entropy")

#call validation curve plotting function.
plot_validation_curve(estimator, DataTrain, TargetTrain, param_name, min_samples_split, addition_graph_points, graph_title, graph_xlabel, graph_ylabel, ylim)	

print("Start of grid search", time.localtime())

# perform GridSearchCV on decision tree of varying min_sample_split
estimator = tree.DecisionTreeClassifier(criterion="entropy")
param_grid = {'min_samples_split':min_samples_split}
grid = GridSearchCV(estimator, param_grid=param_grid, cv=5)
grid.fit(DataTrain,TargetTrain)

print("End of grid search", time.localtime())

print("This is the best score achieved by Decision Tree using GridSearchCV on varying min_sample_split", grid.best_score_)
print("This is the best parameters that achieved the best scores on the Decision Tree using GridSearchCV on varying min_sample_split", grid.best_params_)

clf_dt = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=grid.best_params_['min_samples_split'])
clf_dt = clf_dt.fit(DataTrain, TargetTrain)
print ("Training accuracy of Decision Tree with best parameter from grid search", clf_dt.score(DataTrain, TargetTrain))
print ("Testing accuracy of Decision Tree with best parameter from grid search", clf_dt.score(DataTest, TargetTest))
print("Exiting Decision Tree Classifier")
print("\n")

#*******************Neural Network Classification******************
print("Entering Neural Network Classifier with time ", time.localtime())

PyBDataTrain_nn = copy.deepcopy(PyBDataTrain)
PyBDataTest_nn = copy.deepcopy(PyBDataTest)

PyBDataTrain_nn._convertToOneOfMany()
PyBDataTest_nn._convertToOneOfMany()

fnn = buildNetwork(PyBDataTrain_nn.indim, 2, PyBDataTrain_nn.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer( fnn, dataset=PyBDataTrain_nn, momentum=0.1, verbose=True, weightdecay=0.01)

epochs = 6
trnerr = []
tsterr = []
for i in xrange(epochs):
	# If you set the 'verbose' trainer flag, this will print the total error as it goes.
	trainer.trainEpochs(3)
	trnresult = percentError(trainer.testOnClassData(), PyBDataTrain_nn['class'])
	tstresult = percentError(trainer.testOnClassData(dataset=PyBDataTest_nn), PyBDataTest_nn['class'])
	print ("epoch: %4d" % trainer.totalepochs, " train error: %5.2f%%" % trnresult, " test error: %5.2f%%" % tstresult)
	trnerr.append(trnresult)
	tsterr.append(tstresult)

fig_nn = plt.figure()
ax = fig_nn.add_subplot(1, 1, 1)
ax.set_title("Neural Network Convergence")
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
ax.semilogy(range(len(trnerr)), trnerr, 'b', range(len(tsterr)), tsterr, 'r')

# Check the accuracy
print ("DEFAULT NEURAL NETWORK")
print ("Training Accuracy: " + str(1 - percentError(trainer.testOnClassData(), PyBDataTrain_nn['class'])/100.0))
print ("Testing Accuracy: " + str(1 - percentError(trainer.testOnClassData(dataset=PyBDataTest_nn), PyBDataTest_nn['class'])/100.0))

print("Exiting Neural Network Classifier with time ", time.localtime())

#*******************K Nearest Neighbour Classification******************
print("Entering KNN Classifier with time ", time.localtime())

neigh = KNeighborsClassifier()
neigh = neigh.fit(DataTrain, TargetTrain)

print("Completed initial fit", time.localtime())

print ("Training accuracy of KNN with default settings", neigh.score(DataTrain, TargetTrain))
print ("Testing accuracy of KNN with default settings", neigh.score(DataTest, TargetTest))

# create plot of KNN learning curve
graph_title = "KNN Learning Curve"
graph_xlabel = "Number of Samples"
graph_ylabel = "Score"
ylim = (.7, 1.1)
estimator = KNeighborsClassifier()
plot_learning_curve(estimator, DataTrain, TargetTrain, graph_title, graph_xlabel, graph_ylabel, ylim)

# create points for validation curve plotting
n_neighbors = np.arange(1, 16, 1)
p = [1,2]

for d in p:
	print("Entering loop for KNN validation curve for p value %d" %d)

	full_train_score = []
	unseen_test_scores = []
	for n in n_neighbors:
		neigh = KNeighborsClassifier(n_neighbors=n, p=d)
		neigh = neigh.fit(DataTrain, TargetTrain)
		full_train_score.append(neigh.score(DataTrain, TargetTrain))
		unseen_test_scores.append(neigh.score(DataTest, TargetTest))

	print("Done gathering all the data points for KNN valiation curve for p value %d" %d)

	# Variable for plotting validation curve
	addition_graph_points = [{'data':full_train_score, 'color':'y', 'label': 'Full Training Data Score'}, 
		{'data':unseen_test_scores, 'color':'g', 'label': 'Full Test Data Score'}]
	if d==1:
		graph_title = "KNN Validation Curve Using Manhattan Distance"
	elif d==2:
		graph_title = "KNN Validation Curve Using Eucidean Distance"
	else:
		graph_title = "KNN Validation Curve"
	
	graph_xlabel = "Number of Nearest Neighbours"
	graph_ylabel = "Score"
	ylim = (.7, 1.1)
	param_name = "n_neighbors"
	estimator = KNeighborsClassifier(p=d)

	#call validation curve plotting function.
	plot_validation_curve(estimator, DataTrain, TargetTrain, param_name, n_neighbors, addition_graph_points, graph_title, graph_xlabel, graph_ylabel, ylim)	

print("Start of grid search", time.localtime())

# perform GridSearchCV
estimator = KNeighborsClassifier()
param_grid = {'n_neighbors':n_neighbors, 'p':p}
grid = GridSearchCV(estimator, param_grid=param_grid, cv=5)
grid.fit(DataTrain,TargetTrain)

print("End of grid search", time.localtime())

print("This is the best score achieved by KNN using GridSearchCV on varying n_neighbors and p values", grid.best_score_)
print("This are the best parameters that achieved the best scores on the KNN using GridSearchCV on varying n_neighbors and p values", grid.best_params_)

neigh = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'], p=grid.best_params_['p'])
neigh = neigh.fit(DataTrain, TargetTrain)
print ("Training accuracy of KNN with best parameter from grid search", neigh.score(DataTrain, TargetTrain))
print ("Testing accuracy of KNN with best parameter from grid search", neigh.score(DataTest, TargetTest))
print("Exiting KNN Classifier")
print("\n")

#*******************Boosting Classification******************
print("Entering Boosting Classifier with time ", time.localtime())

clf_boost = ensemble.AdaBoostClassifier()
clf_boost = clf_boost.fit(DataTrain, TargetTrain)

print("Completed initial fit ", time.localtime())

print ("Training accuracy of Boosting", clf_boost.score(DataTrain, TargetTrain))
print ("Testing accuracy of Boosting", clf_boost.score(DataTest, TargetTest))
print()

# create plot of Boosting learning curve
graph_title = "Boosting Learning Curve"
graph_xlabel = "Number of Samples"
graph_ylabel = "Score"
ylim = (.7, 1.1)
estimator = ensemble.AdaBoostClassifier()
plot_learning_curve(estimator, DataTrain, TargetTrain, graph_title, graph_xlabel, graph_ylabel, ylim)

# Validation curve data points creation
n_estimators = np.arange(50, 1550, 50)
print(n_estimators)
full_train_score = []
unseen_test_scores = []

for m in n_estimators:
	clf_boost = ensemble.AdaBoostClassifier(n_estimators=m)
	clf_boost = clf_boost.fit(DataTrain, TargetTrain)
	full_train_score.append(clf_boost.score(DataTrain, TargetTrain))
	unseen_test_scores.append(clf_boost.score(DataTest, TargetTest))

# Variable for plotting validation curve
addition_graph_points = [{'data':full_train_score, 'color':'y', 'label': 'Full Training Data Score'}, 
	{'data':unseen_test_scores, 'color':'g', 'label': 'Full Test Data Score'}]
graph_title = "AdaBoosting Validation Curve"
graph_xlabel = "Number of Estimators"
graph_ylabel = "Score"
ylim = (.7, 1.1)
param_name = "n_estimators"
estimator = ensemble.AdaBoostClassifier()

#call validation curve plotting function.
plot_validation_curve(estimator, DataTrain, TargetTrain, param_name, n_estimators, addition_graph_points, graph_title, graph_xlabel, graph_ylabel, ylim)	

print("Start of grid search", time.localtime())

# perform GridSearchCV
estimator = ensemble.AdaBoostClassifier()
param_grid = {'n_estimators':n_estimators}
grid = GridSearchCV(estimator, param_grid=param_grid, cv=5)
grid.fit(DataTrain,TargetTrain)

print("End of grid search", time.localtime())

print("This is the best score achieved by AdaBoosting using GridSearchCV on varying n_estimators ", grid.best_score_)
print("This are the best parameters that achieved the best scores on the AdaBoosting using GridSearchCV on varying n_estimators ", grid.best_params_)

clf_boost = ensemble.AdaBoostClassifier(n_estimators=grid.best_params_['n_estimators'])
clf_boost = clf_boost.fit(DataTrain, TargetTrain)

print ("Training accuracy of AdaBoost with best parameter from grid search", clf_boost.score(DataTrain, TargetTrain))
print ("Testing accuracy of AdaBoost with best parameter from grid search", clf_boost.score(DataTest, TargetTest))
print("Exiting Boosting Classifier")
print("\n")

#*******************SVM Classification******************
print("Entering SVM Classifier with time ", time.localtime())

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(Data, Target)

print("Completed initial fit", time.localtime())

print ("Training accuracy of SVM", clf_svm.score(DataTrain, TargetTrain))
print ("Testing accuracy of SVM", clf_svm.score(DataTest, TargetTest))
print()

# create plot of SVM learning curve
graph_title = "SVM Learning Curve"
graph_xlabel = "Number of Samples"
graph_ylabel = "Score"
ylim = (.5, 1.1)
estimator = svm.SVC(kernel='linear')
plot_learning_curve(estimator, DataTrain, TargetTrain, graph_title, graph_xlabel, graph_ylabel, ylim)

clf_svm = svm.SVC(kernel='linear')
scores = cross_validation.cross_val_score(clf_svm, DataTrain, TargetTrain, cv=5)
print("The mean accuracy score of kfold cross validation for SVM with linear kernal is ", sum(scores) / len(scores))

print("Exiting SVM Classifier")

print("\n")

print("End of Machine Learning Program with time", time.localtime())
