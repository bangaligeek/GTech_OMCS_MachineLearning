from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from scipy import dot, argmax
from numpy import random
from random import randint
import math

#************************ Start Functions**************************************************
def testOnClassData_custom(net, dataset=None):
        """Return winner-takes-all classification output on a given dataset.

        If no dataset is given, the dataset passed during Trainer
        initialization is used. If return_targets is set, also return
        corresponding target classes.
        """
        assert (dataset != None),"No dataset was provided for the test on class data custom function"
        out = []
        for i in dataset['input']:
            #net.reset()
            res = net.activate(i)
            out.append(argmax(res))
        return out
#************************End of Functions**************************************************

#************************Start Data Prep********************************************
raw_data = np.genfromtxt('BreastCancerWisconsinDataset_modified.txt', delimiter=",", skip_header=1)
raw_inputs = raw_data[:,0:-1]
raw_target = raw_data[:,9:]

assert (raw_inputs.shape[0] == raw_target.shape[0]),"Inputs count and target count do not match"

all_data = ClassificationDataSet(9, 1, nb_classes=2, class_labels=['Benign','Malignant'])

all_data.setField('input', raw_inputs)
all_data.setField('target', raw_target)
all_data.setField('class', raw_target)

test_data_temp, training_data_temp = all_data.splitWithProportion(0.33)

test_data = ClassificationDataSet(9, 1, nb_classes=2, class_labels=['Benign','Malignant'])
for n in xrange(0, test_data_temp.getLength()):
    test_data.addSample(test_data_temp.getSample(n)[0], test_data_temp.getSample(n)[1])

training_data = ClassificationDataSet(9, 1, nb_classes=2, class_labels=['Benign','Malignant'])
for n in xrange(0, training_data_temp.getLength()):
    training_data.addSample(training_data_temp.getSample(n)[0], training_data_temp.getSample(n)[1])

training_data._convertToOneOfMany()
test_data._convertToOneOfMany()

#********************End of Data Preparation***************************

#********************NN With BackPropagation***************************
fnn_backprop = buildNetwork(training_data.indim, 2, training_data.outdim, bias=True, outclass=SoftmaxLayer)

trainer = BackpropTrainer(fnn_backprop, dataset=training_data, momentum=0.1, verbose=True, weightdecay=0.01)

epochs = 10
trnerr_backprop = []
tsterr_backprop = []
for i in xrange(epochs):
    # If you set the 'verbose' trainer flag, this will print the total error as it goes.
    trainer.trainEpochs(1)

    output_train = trainer.testOnClassData()

    for tr in output_train:
        #print("This is the training output value: ", tr)
        continue

    trnresult = percentError(trainer.testOnClassData(), training_data['class'])
    tstresult = percentError(trainer.testOnClassData(dataset=test_data), test_data['class'])
    print ("epoch: %4d" % trainer.totalepochs, " train error: %5.2f%%" % trnresult, " test error: %5.2f%%" % tstresult)
    trnerr_backprop.append(trnresult)
    tsterr_backprop.append(tstresult)
#*****************End of Backpropagation NN*******************************

#*****************Start of RHC NN*******************************
fnn_rhc = buildNetwork(training_data.indim, 2, training_data.outdim, bias=True, outclass=SoftmaxLayer)

trnresult_initial = percentError(testOnClassData_custom(fnn_rhc, dataset=training_data), training_data['class'])
tstresult_initial = percentError(testOnClassData_custom(fnn_rhc, dataset=test_data), test_data['class'])

initial = {}
winner = {}

initial['trainingError'] = trnresult_initial
initial['testError'] = tstresult_initial
initial['FirstEpoch'] = -1.0
initial['LastEpoch'] = -1.0
initial['w'] = fnn_rhc.params[:]

winner['trainingError'] = initial['trainingError']
winner['testError'] = initial['testError']
winner['FirstEpoch'] = initial['FirstEpoch']
winner['LastEpoch'] = initial['LastEpoch']
winner['w'] = initial['w']

print ("Winning Trainning Error: %5.2f%%" % winner['trainingError'], " Winning Test Error: %5.2f%%" % winner['testError'], " First Epoch For Lowest: %4d" %  winner['FirstEpoch'], "Last Epoch For Lowest: %4d" %  winner['LastEpoch'])

epochs = 10
trnerr_rhc = []
tsterr_rhc = []
CW = []
NWP = [None]*len(fnn_rhc.params)
NWM = [None]*len(fnn_rhc.params)
for i in xrange(epochs):
    for j in range(len(fnn_rhc.params)):
        random.seed()
        delta = random.uniform(-1,1)*.05
        NWP[j] = winner['w'][j]+delta

    fnn_rhc.params[:] = NWP[:]


    trnresult_newP = percentError(testOnClassData_custom(fnn_rhc, dataset=training_data), training_data['class'])
    tstresult_newP = percentError(testOnClassData_custom(fnn_rhc, dataset=test_data), test_data['class'])
    print ("epoch: %4d" %  i, " train error for positive step: %5.2f%%" % trnresult_newP, " test error for positive step: %5.2f%%" % tstresult_newP)

    if(trnresult_newP <= winner['trainingError']):
        if(trnresult_newP != winner['trainingError']):
            winner['FirstEpoch'] = i
        winner['trainingError'] = trnresult_newP
        winner['testError'] = tstresult_newP
        winner['LastEpoch'] = i
        winner['w'] = NWP[:]

    print ("Winning Trainning Error: %5.2f%%" % winner['trainingError'], " Winning Test Error: %5.2f%%" % winner['testError'], " First Epoch For Lowest: %4d" %  winner['FirstEpoch'], "Last Epoch For Lowest: %4d" %  winner['LastEpoch'])
#*****************End of RHC NN*******************************

#*****************Start of SA NN*******************************
fnn_sa = buildNetwork(training_data.indim, 2, training_data.outdim, bias=True, outclass=SoftmaxLayer)

trnresult_initial = percentError(testOnClassData_custom(fnn_sa, dataset=training_data), training_data['class'])
tstresult_initial = percentError(testOnClassData_custom(fnn_sa, dataset=test_data), test_data['class'])

initial = {}
winner = {}

initial['trainingError'] = trnresult_initial
initial['testError'] = tstresult_initial
initial['FirstEpoch'] = -1.0
initial['LastEpoch'] = -1.0
initial['w'] = fnn_sa.params[:]

winner['trainingError'] = initial['trainingError']
winner['testError'] = initial['testError']
winner['FirstEpoch'] = initial['FirstEpoch']
winner['LastEpoch'] = initial['LastEpoch']
winner['w'] = initial['w']

print ("Winning Trainning Error: %5.2f%%" % winner['trainingError'], " Winning Test Error: %5.2f%%" % winner['testError'], " First Epoch For Lowest: %4d" %  winner['FirstEpoch'], "Last Epoch For Lowest: %4d" %  winner['LastEpoch'])

epochs = 1000
trnerr_sa = []
tsterr_sa = []
CW = []
NWP = [None]*len(fnn_sa.params)
NWM = [None]*len(fnn_sa.params)
T = 1000000
for i in xrange(epochs):
    for j in range(len(fnn_sa.params)):
        random.seed()
        delta = random.uniform(-1,1)*.05
        NWP[j] = winner['w'][j]+delta

    fnn_sa.params[:] = NWP[:]

    trnresult_newP = percentError(testOnClassData_custom(fnn_sa, dataset=training_data), training_data['class'])
    tstresult_newP = percentError(testOnClassData_custom(fnn_sa, dataset=test_data), test_data['class'])
    print ("epoch: %4d" %  i, " train error for positive step: %5.2f%%" % trnresult_newP, " test error for positive step: %5.2f%%" % tstresult_newP)

    if (T != 0):
        p = 1/(1+(pow(math.e,((trnresult_newP-winner['trainingError'])/T))))
    else:
        p = 0
    if((trnresult_newP <= winner['trainingError']) or random.random()< p):
        if(trnresult_newP != winner['trainingError']):
            winner['FirstEpoch'] = i
        winner['trainingError'] = trnresult_newP
        winner['testError'] = tstresult_newP
        winner['LastEpoch'] = i
        winner['w'] = NWP[:]

    T = T*.95
    if (T < .5):
        T = 0
    print ("Temperture: ", T)

    print ("Winning Trainning Error: %5.2f%%" % winner['trainingError'], " Winning Test Error: %5.2f%%" % winner['testError'], " First Epoch For Lowest: %4d" %  winner['FirstEpoch'], "Last Epoch For Lowest: %4d" %  winner['LastEpoch'])
#*****************End of SA NN*******************************

print ("This is the length of the training and test data, respectively", len(training_data), len(test_data))
print (training_data.indim, training_data.outdim)
print ("This is the shape of the input", all_data['input'].shape)
print ("This is the shape of the target", all_data['target'].shape)
print ("This is the shape of the class", all_data['class'].shape)
print ("This is count of classes", all_data.nClasses)
print ("Here is the statistics on the class", all_data.calculateStatistics())
print ("Here the linked fields", all_data.link)
print ("This is the shape of the input in training", training_data['input'].shape)
print ("This is the shape of the target in training", training_data['target'].shape)
print ("This is the shape of the class in training", training_data['class'].shape)
print ("This is the shape of the input in training", test_data['input'].shape)
print ("This is the shape of the target in training", test_data['target'].shape)
print ("This is the shape of the class in training", test_data['class'].shape)