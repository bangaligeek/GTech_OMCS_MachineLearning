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
#import pydot
from sklearn import cross_validation
import pybrain #install using pip install -i https://pypi.binstar.org/pypi/simple pybrain
from pybrain.datasets import ClassificationDataSet

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

print("this is the target label encoded",len(Target))

vec = DictVectorizer()
Data = vec.fit_transform(FormatedRawData).toarray()

DataTrain, DataTest, TargetTrain, TargetTest = train_test_split(Data, Target, test_size=0.33, random_state=0)

print()
print (Data.shape, "Original data create\n")
print (DataTrain.shape, "Training data create\n")
print (DataTest.shape, "Training data create\n")
print()

#prepare data for pybrain
number_of_columns = Data.shape[1]
PyBData = ClassificationDataSet(number_of_columns, 1, nb_classes=2)
print(PyBData)
PyBData.setField('input', Data)
print(PyBData)
print("target shape",Target.shape[0])
PyBData.setField('target', Target)
print("This is the target",PyBData['target'])


#*******************End of Preparing Data & Target for Estimators******************
#*******************Decision Tree Classification******************

clf_dt = tree.DecisionTreeClassifier()
clf_dt = clf_dt.fit(DataTrain, TargetTrain)

print()
print ("Training accuracy of DT", clf_dt.score(DataTrain, TargetTrain))
print ("Testing accuracy of DT", clf_dt.score(DataTest, TargetTest))
print()

#*******************K Nearest Neighbour Classification******************
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(DataTrain, TargetTrain)

print ("Training accuracy of KNN", neigh.score(DataTrain, TargetTrain))
print ("Testing accuracy of KNN", neigh.score(DataTest, TargetTest))
print()

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