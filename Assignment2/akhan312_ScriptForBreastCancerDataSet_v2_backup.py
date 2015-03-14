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


#training_data, test_data = all_data.splitWithProportion(0.67)

training_data._convertToOneOfMany()
test_data._convertToOneOfMany()

print ("training outdim", training_data.outdim)

fnn = buildNetwork(training_data.indim, 2, training_data.outdim, bias=True, outclass=SoftmaxLayer)

print fnn.activate([7,3,2,10,5,10,5,4,4])

trainer = BackpropTrainer(fnn, dataset=training_data, momentum=0.1, verbose=True, weightdecay=0.01)

epochs = 20
trnerr = []
tsterr = []
for i in xrange(epochs):
    # If you set the 'verbose' trainer flag, this will print the total error as it goes.
    trainer.trainEpochs(5)

    output_train = trainer.testOnClassData()

    for tr in output_train:
        continue
        #print("This is the training output value: ", tr)

    trnresult = percentError(trainer.testOnClassData(), training_data['class'])
    tstresult = percentError(trainer.testOnClassData(dataset=test_data), test_data['class'])
    print ("epoch: %4d" % trainer.totalepochs, " train error: %5.2f%%" % trnresult, " test error: %5.2f%%" % tstresult)
    trnerr.append(trnresult)
    tsterr.append(tstresult)

fig_nn = plt.figure()
ax = fig_nn.add_subplot(1, 1, 1)
ax.set_title("Neural Network Convergence")
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
ax.semilogy(range(len(trnerr)), trnerr, 'b', range(len(tsterr)), tsterr, 'r')

print fnn.activate([7,3,2,10,5,10,5,4,4])
print ("This is the length of the training and test data, respectively", len(training_data), len(test_data))
print (training_data.indim, training_data.outdim)
print ("This is the shape of the input", all_data['input'].shape)
print ("This is the shape of the target", all_data['target'].shape)
print ("This is the shape of the class", all_data['class'].shape)
print ("This is count of classes", all_data.nClasses)
print ("Here is the statistics on the class", all_data.calculateStatistics())
print ("Here the linked fields", all_data.link)