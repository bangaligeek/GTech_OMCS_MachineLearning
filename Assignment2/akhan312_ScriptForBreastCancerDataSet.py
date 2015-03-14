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

print ("this is the variable statistics ", PyBData.calculateStatistics())
#*******************End of Preparing Data & Target for Estimators******************

#*******************Neural Network Classification******************
print("Entering Neural Network Classifier with time ", time.localtime())

PyBDataTrain_nn = copy.deepcopy(PyBDataTrain)
PyBDataTest_nn = copy.deepcopy(PyBDataTest)

PyBDataTrain_nn._convertToOneOfMany()
PyBDataTest_nn._convertToOneOfMany()

fnn = buildNetwork(PyBDataTrain_nn.indim, 2, PyBDataTrain_nn.outdim, bias=True, outclass=SoftmaxLayer)
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

#******************* End ******************