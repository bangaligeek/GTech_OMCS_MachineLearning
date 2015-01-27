from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
import numpy as np
import csv


data = list(csv.DictReader(open('income_data.txt', 'rU')))

print(data)
print()

Y = []

for x in data:
	Y.append(x['l'])
	del x['l']
	x['a'] = float(x['a'])
	x['c'] = float(x['c'])

print(data)
print()

print(Y)
print()

Y = np.array(Y)

le = preprocessing.LabelEncoder()
le.fit(Y)

print(Y)
print()

target = np.array(Y)

print(target)
print()

target = le.transform(target)

print(target)
print()

vec = DictVectorizer()
data = vec.fit_transform(data).toarray()

print(data)
print()

print ('Vectorized:', data[0])
print ('Unvectorized:', vec.inverse_transform(data[0]))


clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, target)

predict_data = [0.1292, 0., 1., 0., 0., 0.5929]
predicted_value = clf.predict(predict_data)

print ('Unvectorized Predict Data:', vec.inverse_transform(predict_data))

print("predicted value is", le.inverse_transform(predicted_value))
