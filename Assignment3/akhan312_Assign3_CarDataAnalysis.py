import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from scipy.stats import kurtosis
from sklearn.ensemble import ExtraTreesClassifier

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

class customDataset:

	def __init__(self,dataFile, targetHeaderName='Class', delimiter=',', header=0):
             # Read the data from CSV file
             self.rawData = pd.read_csv(dataFile, sep=delimiter, header=header)
             # Convert the raw data read into dict structure after removing the target column
             self.rawDataDict = self.rawData.drop([targetHeaderName], axis=1).T.to_dict().values()
             
             # Extract the target column from raw data into raw target variable
             rawTarget_temp = np.array(self.rawData[targetHeaderName])
             self.rawTarget = np.reshape(rawTarget_temp,(len(rawTarget_temp),1))
             
             # Vectorize the raw data and transform it inputs array
             self.vec = DictVectorizer()
             self.inputs = self.vec.fit_transform(self.rawDataDict).toarray()        
          
             # Encode the raw target column so the targets are labeled start 0 to n where n is number of classes
             self.le = preprocessing.LabelEncoder()
             self.le.fit(self.rawTarget)  
             self.target = self.le.transform(self.rawTarget)

             # split data into training and test sets
             self.dataTrain, self.dataTest, self.targetTrain, self.targetTest = train_test_split(self.inputs, self.target, test_size=0.33, random_state=0)
             
             #self.dataTrainScaled = self.dataTrain
             #self.dataTestScaled = self.dataTest
             
             self.dataScaler = preprocessing.StandardScaler().fit(self.dataTrain)
             self.dataTrainScaled = self.dataScaler.transform(self.dataTrain)
             self.dataTestScaled = self.dataScaler.transform(self.dataTest)
             
             #self.dataScalerMeanOnly = preprocessing.StandardScaler(with_std=False).fit(self.dataTrain)
             #self.dataTrainScaledMeanOnly = self.dataScaler.transform(self.dataTrain)
             #self.dataTestScaledMeanOnly = self.dataScaler.transform(self.dataTest)

             # Normalize data
             #self.nz = preprocessing.Normalizer().fit(self.inputsUnNorm)
             #self.inputs = self.nz.transform(self.inputsUnNorm)
             
             #self.scaler = preprocessing.StandardScaler().fit(self.inputsUnNorm)
             #self.inputs = self.scaler.transform(self.inputsUnNorm)
             
             # split data into training and test sets
             #self.dataTrain, self.dataTest, self.targetTrain, self.targetTest = train_test_split(self.inputs, self.target, test_size=0.33, random_state=0)

def plot_curve(graphData, graph_title, graph_xlabel, graph_ylabel, ylim=None):
    plt.figure()
    plt.title(graph_title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(graph_xlabel)
    plt.ylabel(graph_ylabel)
        
    plt.grid()
    
    for d in graphData:
       # varianceMean = np.mean(d[1])
        ySTD = np.std(d[1])
        plt.fill_between(d[0], d[1] - ySTD,d[1] + ySTD, alpha=0.1,color=d[3])
                         
        plt.plot(d[0], d[1], 'o-', color=d[3], label=d[2])

    plt.legend(loc="best")
    plt.savefig('plots/carPlots/'+graph_title+'.png')
    plt.close()
    plt.show()
    
def plot_scatter2D(graphData, Y, ylim=None):
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    
    # Plot the training points
    plt.scatter(graphData[:, 0], graphData[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('1st eigenvector')
    plt.ylabel('2nd eigenvector')

    plt.xticks(())
    plt.yticks(())
    
def plot_scatter3D(graphData, Y, ylim=None):
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(graphData[:, 0], graphData[:, 1], graphData[:, 2], c=Y, cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()

def plot_learning_curve(x, training_erorr, test_error, graph_title, graph_xlabel, graph_ylabel, ylim=None):
    
    plt.figure()
    plt.title(graph_title)
    if ylim is not None:
        plt.ylim(*ylim)
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

    plt.plot(x, training_erorr, 'o-', color="r", label="Training score")
    plt.plot(x, test_error, 'o-', color="g", label="Test Score")

    plt.legend(loc="best")
    plt.savefig('plots/carPlots/'+graph_title+'.png')
    plt.close()
    #plt.show()
    
def kMeansScore(data, allKs=[1], datasetType=None, target=None):
    # K-MEANS
    km = KMeans()
    ks = []
    kmScore = []
    f = open('plots/carPlots/KMeanClusterStats_'+datasetType+'.txt', 'w')
    targetLabels, targetStats = np.unique(target, return_counts=True)
    f.write("Data Target Labels: "+str(targetLabels)+"\n")
    f.write("Data Target Stats: "+str(targetStats)+"\n\n\n")
    for k in allKs:
        km.set_params(n_clusters=k)
        ks.append(k)
        km.fit(data)
        kmScore.append(-km.inertia_)
        labels, stats = np.unique(km.labels_, return_counts=True)        
        f.write("Cluster Stats For K = "+str(k)+"\n")
        f.write("Unique Labels: "+str(labels)+"\n")
        f.write("Status Corresponding To The Labels: "+str(stats)+"\n\n")
    
    f.close()
    return ks, kmScore 

def ICAkurtosis(data):
    k=[]
    print data.shape
    print range(data.shape[1])
    for d in range(data.shape[1]):
        print kurtosis(data[:,d:d+1])

def EMscore(data, allKs=[1]):
    # K-MEANS
    em = mixture.GMM()
    ks = []
    lgLikelyhood = []
    bicScore = []
    for k in allKs:
        em.set_params(n_components=k)
        ks.append(k)
        em.fit(data)
        lgLikelyhood.append(sum(em.score(data)))
        bicScore.append(em.bic(data))
    return ks, lgLikelyhood, bicScore

def NNBackPropCustom(trainInputs, trainTarget, testInputs, testTarget, inputDim, targetDim, numClass, classLabels, bias=True, numHiddenLayers=2, numEpoch=10, momentum=0.1, weightdecay=0.01):
    #NN Data Preparation
    assert (trainInputs.shape[0] == trainTarget.shape[0]),"Inputs count and target count for your training data do not match for NN Analysis"
    assert (testInputs.shape[0] == testTarget.shape[0]),"Inputs count and target count for your test data do not match for NN Analysis"
    
    training_data = ClassificationDataSet(inputDim, targetDim, nb_classes=numClass, class_labels=classLabels)
    test_data = ClassificationDataSet(inputDim, targetDim, nb_classes=numClass, class_labels=classLabels)
    
    training_data.setField('input', trainInputs)
    training_data.setField('target', trainTarget)
    training_data.setField('class', trainTarget)
    
    test_data.setField('input', testInputs)
    test_data.setField('target', testTarget)
    test_data.setField('class', testTarget)
    
    training_data._convertToOneOfMany()
    test_data._convertToOneOfMany()
    
    # NN With BackPropagation
    fnn_backprop = buildNetwork(training_data.indim, numHiddenLayers, training_data.outdim, bias=bias, outclass=SoftmaxLayer)
    
    trainer = BackpropTrainer(fnn_backprop, dataset=training_data, momentum=momentum, verbose=True, weightdecay=weightdecay)
    
    epochs = numEpoch
    epoch_v = []
    trnerr_backprop = []
    tsterr_backprop = []
    for i in xrange(epochs):
        # If you set the 'verbose' trainer flag, this will print the total error as it goes.
        trainer.trainEpochs(1)
    
        trnresult = percentError(trainer.testOnClassData(), training_data['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=test_data), test_data['class'])
        print ("epoch: %4d" % trainer.totalepochs, " train error: %5.2f%%" % trnresult, " test error: %5.2f%%" % tstresult)
        epoch_v.append(trainer.totalepochs)
        trnerr_backprop.append(trnresult)
        tsterr_backprop.append(tstresult)
    
    return epoch_v, trnerr_backprop, tsterr_backprop
#**********************************End of Functions****************************

# Load data
dataSet = customDataset("CarQualityDataset.txt")

# K-MEANS
ks, kmScore = kMeansScore(dataSet.dataTrainScaled, range(1,10,1), "originalData", dataSet.targetTrain)

# Plot K-MEANS score before any dimensionality transformation
plot_curve([(ks, kmScore, "K-Means w/o Feature Reduction", "r")], 
            graph_title="K-Means Variance", 
            graph_xlabel="K-Means", graph_ylabel="K-Means Variance without Feature Reduction", ylim=None)

# Expectation Maximization
ksEM, lgLikelyhood, bicScore = EMscore(dataSet.dataTrainScaled, range(1,10,1))

# Plot EM log likelyhood score before any dimensionality transformation
plot_curve([(ksEM, lgLikelyhood, "EM w/o Feature Reduction", "r")], 
            graph_title="EM Log Likelyhood", 
            graph_xlabel="# Guassians", graph_ylabel="Log Likelyhood", ylim=None)
            
# Plot EM BIC score before any dimensionality transformation
plot_curve([(ksEM, bicScore, "EM w/o Feature Reduction", "r")], 
            graph_title="EM BIC Score", 
            graph_xlabel="# Guassians", graph_ylabel="BIC", ylim=None)

kmOptimal = KMeans(n_clusters=2)
dataTrainScaledKM = kmOptimal.fit_transform(dataSet.dataTrainScaled)
dataTestScaledKM = kmOptimal.fit_transform(dataSet.dataTestScaled)

# PCA dimensionality transformation for all dimension
pca = PCA(n_components=dataSet.dataTrainScaled.shape[1], whiten=False)
pca.fit(dataSet.dataTrainScaled)

# Plot PCA variance explained by each component, where n_compoents equals number of features
plot_curve([(range(1,dataSet.dataTrainScaled.shape[1]+1,1), pca.explained_variance_ratio_,"PCA Variance", "r")], 
            graph_title="PCA Variance", 
            graph_xlabel="PCA", graph_ylabel="PCA Variance", ylim=None)


# PCA dimensionality transformation with optimal components
pcaOptimal = PCA(n_components=2, whiten=False)
dataTrainScaledPCAoptimal = pcaOptimal.fit_transform(dataSet.dataTrainScaled)

# K-MEANS
ksPCAoptimal, kmScorePCAoptimal = kMeansScore(dataTrainScaledPCAoptimal, range(1,10,1), "PCAOptimalData", dataSet.targetTrain)

# Plot K-MEANS score After dimensionality transformation
plot_curve([(ks, kmScore, "K-Means w/o Feature Reduction", "r"), 
            (ksPCAoptimal, kmScorePCAoptimal, "K-Means With Two Pricipal Components", "g")], 
            graph_title="K-Means Variance with PCA Optimal", 
            graph_xlabel="K-Means", graph_ylabel="K-Means Variance", ylim=None)

#plot_scatter2D(dataTrainScaledPCAoptimal, dataSet.targetTrain, ylim=None)

# Expectation Maximization
ksEMPCA, lgLikelyhoodPCA, bicScorePCA = EMscore(dataTrainScaledPCAoptimal, range(1,10,1))

# Plot EM log likelyhood score before any dimensionality transformation
plot_curve([(ksEMPCA, lgLikelyhoodPCA, "EM with With Two Pricipal Components", "r")], 
            graph_title="EM Log Likelyhood with PCA Optimal", 
            graph_xlabel="# Guassians", graph_ylabel="Log Likelyhood", ylim=None)
            
# Plot EM BIC score before any dimensionality transformation
plot_curve([(ksEMPCA, bicScorePCA, "EM with With Two Pricipal Components", "r")], 
            graph_title="EM BIC Score with PCA Optimal", 
            graph_xlabel="# Guassians", graph_ylabel="BIC", ylim=None)


# PCA dimensionality transformation
pca3Comp = PCA(n_components=3, whiten=False)
dataTrainScaledPCA3Comp = pca3Comp.fit_transform(dataSet.dataTrainScaled)

# K-MEANS
ksPCA3Comp, kmScorePCA3Comp = kMeansScore(dataTrainScaledPCA3Comp, range(1,10,1), "PCA3CompData", dataSet.targetTrain)

# Plot K-MEANS score After dimensionality transformation
plot_curve([(ks, kmScore, "K-Means w/o Feature Reduction", "r"), 
            (ksPCAoptimal, kmScorePCAoptimal, "K-Means With Two Pricipal Components", "g"), 
            (ksPCA3Comp, kmScorePCA3Comp, "K-Means With Three Pricipal Components", "b")], 
            graph_title="K-Means Variance with PCA Three Components", 
            graph_xlabel="K-Means", graph_ylabel="K-Means Variance", ylim=None)

#plot_scatter3D(dataTrainScaledPCA3Comp, dataSet.targetTrain, ylim=None)


for i in range(1,dataSet.dataTrainScaled.shape[1]+1,1):

    # ICA dimensionality transformation
    ica = FastICA(n_components=i, whiten=True)
    dataTrainScaledICA = ica.fit_transform(dataSet.dataTrainScaled)
    
    # K-MEANS
    ksICA, kmScoreICA = kMeansScore(dataTrainScaledICA, range(1,10,1), "ICA"+str(i)+"compData", dataSet.targetTrain)
    
    # Plot K-MEANS score After dimensionality transformation
    plot_curve([(ksICA, kmScoreICA, "K-Means With ICA "+str(i), "y")], 
                graph_title="K-Means Variance with ICA Components", 
                graph_xlabel="K-Means", graph_ylabel="K-Means Variance", ylim=None)
    
    # Expectation Maximization
    ksEMICA, lgLikelyhoodICA, bicScoreICA = EMscore(dataTrainScaledICA, range(1,10,1))
    
    # Plot EM log likelyhood score before any dimensionality transformation
    plot_curve([(ksEMICA, lgLikelyhoodICA, "EM with With ICA "+str(i), "r")], 
                graph_title="EM Log Likelyhood with ICA", 
                graph_xlabel="# Guassians", graph_ylabel="Log Likelyhood", ylim=None)
                
    # Plot EM BIC score before any dimensionality transformation
    plot_curve([(ksEMICA, bicScoreICA, "EM with With ICA "+str(i), "r")], 
                graph_title="EM BIC Score with ICA", 
                graph_xlabel="# Guassians", graph_ylabel="BIC", ylim=None)


for i in range(1,dataSet.dataTrainScaled.shape[1]+1,1):
    # Random Projection dimensionality transformation
    randomProj = random_projection.GaussianRandomProjection(n_components=i, eps=.99)
    dataTrainScaledRP = randomProj.fit_transform(dataSet.dataTrainScaled)
    
    # K-MEANS
    ksRP, kmScoreRP = kMeansScore(dataTrainScaledRP, range(1,10,1), "RP"+str(i)+"compData", dataSet.targetTrain)
    
    # Plot K-MEANS score After dimensionality transformation
    plot_curve([(ksRP, kmScoreRP, "K-Means With Random Projection "+str(i), "c")], 
                graph_title="K-Means Variance with RP", 
                graph_xlabel="K-Means", graph_ylabel="K-Means Variance", ylim=None)
    
    # Expectation Maximization
    ksEMRP, lgLikelyhoodRP, bicScoreRP = EMscore(dataTrainScaledRP, range(1,10,1))
    
    # Plot EM log likelyhood score before any dimensionality transformation
    plot_curve([(ksEMRP, lgLikelyhoodRP, "EM With Random Projection "+str(i), "r")], 
                graph_title="EM Log Likelyhood With RP", 
                graph_xlabel="# Guassians", graph_ylabel="Log Likelyhood", ylim=None)
                
    # Plot EM BIC score before any dimensionality transformation
    plot_curve([(ksEMRP, bicScoreRP, "EM With Random Projection "+str(i), "r")], 
                graph_title="EM BIC Score With RP", 
                graph_xlabel="# Guassians", graph_ylabel="BIC", ylim=None)

# Feature Extraction Via Decision Tree
clf = ExtraTreesClassifier()
dataTrainScaledFSDT = clf.fit(dataSet.dataTrainScaled, dataSet.targetTrain).transform(dataSet.dataTrainScaled)

# K-MEANS
ksFTDT, kmScoreFTDT = kMeansScore(dataTrainScaledFSDT, range(1,10,1), "DTFSData", dataSet.targetTrain)

# Plot K-MEANS score After dimensionality transformation
plot_curve([(ksFTDT, kmScoreFTDT, "K-Means With Decision Tree Feature Reduction", "c")], 
            graph_title="K-Means Variance With Decision Tree Feature Reduction", 
            graph_xlabel="K-Means", graph_ylabel="K-Means Variance", ylim=None)

# Expectation Maximization
ksEMFTDT, lgLikelyhoodFTDT, bicScoreFTDT = EMscore(dataTrainScaledFSDT, range(1,10,1))

# Plot EM log likelyhood score before any dimensionality transformation
plot_curve([(ksEMFTDT, lgLikelyhoodFTDT, "EM With Decision Tree Feature Reduction", "r")], 
            graph_title="EM Log Likelyhood With Decision Tree Feature Reduction", 
            graph_xlabel="# Guassians", graph_ylabel="Log Likelyhood", ylim=None)
            
# Plot EM BIC score before any dimensionality transformation
plot_curve([(ksEMFTDT, bicScoreFTDT, "EM With Decision Tree Feature Reduction", "r")], 
            graph_title="EM BIC Score With Decision Tree Feature Reduction", 
            graph_xlabel="# Guassians", graph_ylabel="BIC", ylim=None)

