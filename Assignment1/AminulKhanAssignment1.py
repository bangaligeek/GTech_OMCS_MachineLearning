from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import csv

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
#*******************End of Preparing Data & Target for Estimators******************

Predict_Data = Data[26614] 

clf_dt = tree.DecisionTreeClassifier()
clf_dt = clf_dt.fit(Data, Target)

predict_target = clf_dt.predict(Predict_Data)

print("This is the predicted value using decision tree", predict_target)
print()

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(Data, Target)

predict_target = neigh.predict(Predict_Data)
print("This is the predicted value using KNN", predict_target)
print()

clf_svm = svm.SVC()
clf_svm.fit(Data, Target)

predict_target = clf_svm.predict(Predict_Data)

print("This is the predicted value using SVM", predict_target)
print()

