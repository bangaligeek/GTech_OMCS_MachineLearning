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

#*******************All Functions*****************

# this function takes a input of dict of data with as InputData, keys to values that need to converted from string to float, and key for the classification field
# The function returns InputData in the same dict form but number values changed from string to float and with classification data removed.
# The function also returns all the classification values in the list called ExtractedTarget
def extract_target_ConvertStringNumber(InputData, numStringKeys, classKey):
	for d in InputData:
		ExtractedTarget.append(d[classKey])
		del d[classKey]
		for k in numStringKeys:
			d[k] = float(d[k])
		d['age'] = float(d['age'])
		d['fnlwgt'] = float(d['fnlwgt'])
		d['education-num'] = float(d['education-num'])
		d['capital-gain'] = float(d['capital-gain'])
		d['capital-loss'] = float(d['capital-loss'])
		d['hours-per-week'] = float(d['hours-per-week'])
	
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

def plot_validation_curve(estimator, X, y, param_name, param_range, addition_graph_points, graph_title, graph_xlabel, graph_ylabel, ylim):
	
	cv_train_scores, cv_test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=3, scoring="accuracy")

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
	plt.show()

#*******************End of Functions**************
#*******************Prepare Data & Target for Estimators******************
InputData = list(csv.DictReader(open('income_data_withheader.txt', 'rU')))

numStringKeys = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
classKey = 'class'
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
'''
clf_dt = tree.DecisionTreeClassifier(criterion="entropy")
clf_dt = clf_dt.fit(DataTrain, TargetTrain)

print()
print ("Training accuracy of Decision Tree with default settings and criterion=entropy", clf_dt.score(DataTrain, TargetTrain))
print ("Testing accuracy of Decision Tree with default settings and criterion=entropy", clf_dt.score(DataTest, TargetTest))
print()

# create points for validation curve plotting of the accuracy score of full Training and Test data on varying values of min_sample_split
min_samples_split = np.linspace(50, 5000, 100)
full_train_score = []
unseen_test_scores = []

for m in min_samples_split:
	clf_dt = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=m)
	clf_dt = clf_dt.fit(DataTrain, TargetTrain)
	full_train_score.append(clf_dt.score(DataTrain, TargetTrain))
	unseen_test_scores.append(clf_dt.score(DataTest, TargetTest))

# Variable for plotting validation curve
addition_graph_points = [{'data':full_train_score, 'color':'y', 'label': 'Full Training Data Score'}, 
	{'data':unseen_test_scores, 'color':'g', 'label': 'Full Training Data Score'}]
graph_title = "Decision Tree Validation Curve"
graph_xlabel = "Min Samples Split"
graph_ylabel = "Score"
ylim = (.7, 1.1)
param_name = "min_samples_split"
estimator = tree.DecisionTreeClassifier(criterion="entropy")

#call validation curve plotting function.
plot_validation_curve(estimator, DataTrain, TargetTrain, param_name, min_samples_split, addition_graph_points, graph_title, graph_xlabel, graph_ylabel, ylim)	


# perform GridSearchCV on decision tree of varying min_sample_split
param_grid = {'min_samples_split':min_samples_split}
grid = GridSearchCV(clf_dt, param_grid=param_grid, cv=5)
grid.fit(DataTrain,TargetTrain)

print("This is the best score achieved by Decision Tree using GridSearchCV on varying min_sample_split", grid.best_score_)
print("This is the best parameters that achieved the best scores on the Decision Tree using GridSearchCV on varying min_sample_split", grid.best_params_)

#*******************Neural Network Classification******************
PyBDataTrain_nn = copy.deepcopy(PyBDataTrain)
PyBDataTest_nn = copy.deepcopy(PyBDataTest)

PyBDataTrain_nn._convertToOneOfMany()
PyBDataTest_nn._convertToOneOfMany()

fnn = buildNetwork(PyBDataTrain_nn.indim, 5, PyBDataTrain_nn.outdim, outclass=SoftmaxLayer)
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
print ("\n" + "*" * 50)
print ("DEFAULT NEURAL NETWORK")
print ("Training Accuracy: " + str(1 - percentError(trainer.testOnClassData(), PyBDataTrain_nn['class'])/100.0))
print ("Testing Accuracy: " + str(1 - percentError(trainer.testOnClassData(dataset=PyBDataTest_nn), PyBDataTest_nn['class'])/100.0))
'''
#*******************K Nearest Neighbour Classification******************
neigh = KNeighborsClassifier()
neigh.fit(DataTrain, TargetTrain)

print ("Training accuracy of KNN", neigh.score(DataTrain, TargetTrain))
print ("Testing accuracy of KNN", neigh.score(DataTest, TargetTest))
print()

# create points for validation curve plotting of the accuracy score of full Training and Test data on varying values of min_sample_split
n_neighbors = [1, 2, 3, 4, 5]
p = [1,2]
full_train_score = []
unseen_test_scores = []

for d in p:
	for n in n_neighbors:
		neigh = KNeighborsClassifier(n_neighbors=n, p=d)
		neigh = neigh.fit(DataTrain, TargetTrain)
		full_train_score.append(neigh.score(DataTrain, TargetTrain))
		unseen_test_scores.append(neigh.score(DataTest, TargetTest))
		print("data created for knn = ", n)

	# Variable for plotting validation curve
	addition_graph_points = [{'data':full_train_score, 'color':'y', 'label': 'Full Training Data Score'}, 
		{'data':unseen_test_scores, 'color':'g', 'label': 'Full Training Data Score'}]
	graph_title = "KNN Validation Curve"
	graph_xlabel = "Number of Nearest Neighbours"
	graph_ylabel = "Score"
	ylim = (.7, 1.1)
	param_name = "n_neighbors"
	estimator = KNeighborsClassifier(p=d)

	#call validation curve plotting function.
	plot_validation_curve(estimator, DataTrain, TargetTrain, param_name, n_neighbors, addition_graph_points, graph_title, graph_xlabel, graph_ylabel, ylim)	



#*******************Boosting Classification******************
clf_boost = ensemble.AdaBoostClassifier()
clf_boost = clf_boost.fit(DataTrain, TargetTrain)

print ("Training accuracy of Boosting", clf_boost.score(DataTrain, TargetTrain))
print ("Testing accuracy of Boosting", clf_boost.score(DataTest, TargetTest))
print()

#*******************SVM Classification******************
clf_svm = svm.SVC()
clf_svm.fit(Data, Target)

print ("Training accuracy of SVM", clf_svm.score(DataTrain, TargetTrain))
print ("Testing accuracy of SVM", clf_svm.score(DataTest, TargetTest))
print()

'''
#*****************Cross-Validation*******************
scores = cross_validation.cross_val_score(clf_dt, DataTrain, TargetTrain, cv=5)
print(scores)
print()
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()
#print ("Testing accuracy of DT with CV", clf_dt.score(DataTest, TargetTest))
predict_target = clf_dt.predict(Predict_Data)
print("This is the predicted value using decision tree cross validation", predict_target)
print()
print()

#*****************End Cross-Validation***************
'''