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

#*******************All Functions*****************
def extract_target_ConvertStringNumber(InputData):
	for d in InputData:
		ExtractedTarget.append(d['class'])
		del d['class']
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

ExtractedTarget = []

FormatedRawData, ExtractedTarget = extract_target_ConvertStringNumber(InputData)

ExtractedTarget_array = np.array(ExtractedTarget)

le = preprocessing.LabelEncoder()
le.fit(ExtractedTarget_array)

Target = le.transform(ExtractedTarget_array)

vec = DictVectorizer()
Data = vec.fit_transform(FormatedRawData).toarray()

DataTrain, DataTest, TargetTrain, TargetTest = train_test_split(Data, Target, test_size=0.33, random_state=0)

#*******************End of Preparing Data & Target for Estimators******************

print (Data.shape, "Original data create\n")
print (DataTrain.shape, "Training data create\n")
print (DataTest.shape, "Training data create\n")

Predict_Data = Data[26614] 

clf_dt = tree.DecisionTreeClassifier()
#clf_dt = clf_dt.fit(DataTrain, TargetTrain)

'''
dot_data = StringIO() 
tree.export_graphviz(clf_dt, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("decisionTree.pdf") 
'''
'''
print()
print()
print ("Training accuracy of DT", clf_dt.score(DataTrain, TargetTrain))
print ("Testing accuracy of DT", clf_dt.score(DataTest, TargetTest))

predict_target = clf_dt.predict(Predict_Data)

print("This is the predicted value using decision tree", predict_target)
print()
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

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(Data, Target)

print ("Training accuracy of KNN", neigh.score(DataTrain, TargetTrain))
print ("Testing accuracy of KNN", neigh.score(DataTest, TargetTest))

predict_target = neigh.predict(Predict_Data)
print("This is the predicted value using KNN", predict_target)
print()
print()


'''
clf_boost = ensemble.AdaBoostClassifier()
clf_boost = clf_boost.fit(Data, Data)

predict_target = clf_boost.predict(Predict_Data)
print("This is the predicted value using Boosting", predict_target)
print()

clf_svm = svm.SVC()
clf_svm.fit(Data, Target)

predict_target = clf_svm.predict(Predict_Data)

print("This is the predicted value using SVM", predict_target)
print()
'''

