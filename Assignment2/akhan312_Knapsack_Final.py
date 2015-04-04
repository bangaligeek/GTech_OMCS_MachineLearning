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

def geneticoptimize(maxiter,domain,net,costf,popsize=100,step=.05, mutprob=0.2,elite=0.2):
    # Mutation Operation
    def mutate(vec):
        i=random.randint(0,len(domain)-1)
        if random.random( )<0.5 and vec[i]>domain[i][0]:
            return vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif vec[i]<domain[i][1]:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]

    # Crossover Operation
    def crossover(r1,r2):
        i=random.randint(1,len(domain)-2)
        return r1[0:i]+r2[i:]
    
    # Build the initial population
    pop=[]
    for i in range(popsize):
        vec=[random.uniform(domain[i][0],domain[i][1]) for i in range(len(domain))]
        pop.append(vec)

    # How many winners from each generation?
    topelite=int(elite*popsize)
    
    # Main loop
    scores = [None]*len(pop)
    for i in range(maxiter):
        for v in range(len(pop)):
            print len(pop)
            net.params[:]=pop[v][:]
            scores[v] = (costf(net),pop[v][:])
        #scores=[(costf(v),v) for v in pop]
        scores.sort()
        ranked=[v for (s,v) in scores]
        
        # Start with the pure winners
        pop=ranked[0:topelite]
        
        
        # Add mutated and bred forms of the winners
        while len(pop)<popsize:
            if random.random( )<mutprob:
                # Mutation
                c=random.randint(0,topelite)
                pop.append(mutate(ranked[c]))
            else:
                # Crossover
                c1=random.randint(0,topelite)
                c2=random.randint(0,topelite)
                pop.append(crossover(ranked[c1],ranked[c2]))
        
        # Print current best score
        print scores[0][0]

    return scores[0][1]

def hillclimb(domain,costf):
    # Create a random solution
    sol=[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
    
    # Main loop
    while 1:
        # Create list of neighboring solutions
        neighbors=[]
        for j in range(len(domain)):
            # One away in each direction
            if sol[j]>domain[j][0]:
            neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
            if sol[j]<domain[j][1]:
            neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])

        # See what the best solution amongst the neighbors is
        current=costf(sol)
        best=current
        for j in range(len(neighbors)):
            cost=costf(neighbors[j])
            if cost<best:
                best=cost
                sol=neighbors[j]
            # If there's no improvement, then we've reached the top
            if best==current:
                break
    return sol

def plot_learning_curve(x, training_erorr, test_error, graph_title, graph_xlabel, graph_ylabel, ylim=None, xlim=None):
    
    plt.figure()
    plt.title(graph_title)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel(graph_xlabel)
    plt.ylabel(graph_ylabel)

    train_error_mean = np.mean(training_erorr)
    train_error_std = np.std(training_erorr)
    test_error_mean = np.mean(test_error)
    test_error_std = np.std(test_error)

    plt.grid()

    plt.fill_between(x, training_erorr - train_error_std,
                     training_erorr + train_error_std, alpha=0.1,
                     color="r")
    plt.fill_between(x, test_error - test_error_std,
                     test_error + test_error_std, alpha=0.1, color="g")
    print x
    print train_error_mean
    print training_erorr
    plt.plot(x, training_erorr, 'o-', color="r", label="Training score")
    plt.plot(x, test_error, 'o-', color="g", label="Test Score")

    plt.legend(loc="best")
    plt.savefig('plots/'+graph_title+'.png')
    plt.close()
    #plt.show()
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

#********************NN With GA***************************
def fitFunction (net, dataset=training_data, targetClass=training_data['class']):
    error = percentError(testOnClassData_custom(net, dataset=training_data), targetClass)
    return error

stepSize = [.05, .5, 1]
for s in stepSize:
    fnn_ga = buildNetwork(training_data.indim, 2, training_data.outdim, bias=True, outclass=SoftmaxLayer)

    domain = [(-1,1)]*len(fnn_ga.params)
    #print domain
    epochs = 20
    epoch_v = []
    trnerr_ga = []
    tsterr_ga = []
    iteration = 5
    for i in xrange(epochs):
        winner = geneticoptimize(iteration,domain,fnn_ga,fitFunction,popsize=100,step=s, mutprob=0.2,elite=0.2)
        fnn_ga.params[:] = winner[:]
        training_error = fitFunction(fnn_ga, dataset=training_data, targetClass=training_data['class'])
        test_error = fitFunction(fnn_ga, dataset=test_data, targetClass=test_data['class'])
        epoch_v.append(i*iteration)
        trnerr_ga.append(training_error)
        tsterr_ga.append(test_error)
        print ("This is the training and test error at the epoch: ", training_error, test_error, i*iteration)


    ylim = (0, 70)
    xlim = (50, 1005)
    print ("This is epoch_value",epoch_v)
    print ("This is training ga",trnerr_ga)
    print ("This is test ga",tsterr_ga)
    plot_learning_curve(epoch_v, trnerr_ga, tsterr_ga, "Neural Network With GA_step_"+str(s), "Epochs", "Error %", ylim, xlim=None)

#*****************End of GA NN*******************************

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