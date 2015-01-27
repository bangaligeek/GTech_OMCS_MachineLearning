from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
import numpy as np
import csv

dataInput = list(csv.DictReader(open('income_data.txt', 'rU')))

print(dataInput)
print()


class PrepData:
	Data = []
	Target = []

	ExtractedTargetList = []

	ConvertedDataDictList = []

	le = preprocessing.LabelEncoder()

	vec = DictVectorizer()

	def __init__(self):
		print ("Class PrepData has been initialized\n")

	def GetConvertedDataDictList ():
		return ConvertedDataDictList

	def GetExtractedTargetList ():
		return ExtractedTargetList

	def GetData ():
		return Data

	def GetTarget ():
		return Target

	def ConvertDictStringValuesToFloat_ExtractTarget (RawDataDictList, FieldIndicesForStrToFloat, TargetIndex = "none"):

		ConvertedDataDictList = RawDataDictList

		print (self.RawDataDictList)

		for d in ConvertedDataDictList:
			for f in FieldIndices:
				d[f] = float(d[f])

			if TargetIndex != "none":
				ExtractedTargetList.append(d[TargetIndex])
				del d[TargetIndex]

	def EncodeStringCategoryTargets(TargetList):
		t = np.array(TargetList)
		le.fit(t)

		target = le.transform(t)

	def VectorizeStringFeatures(DataDictList):
		data = vec.fit_transform(data).toarray()



x = PrepData()

i = ['a','c']
for d in dataInput:
	for f in i:
		d[f] = float(d[f])

print(dataInput)
print()

x.ConvertDictStringValuesToFloat_ExtractTarget(dataInput)

print (GetExtractedTargetList())



