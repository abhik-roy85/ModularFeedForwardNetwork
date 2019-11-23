import os
import numpy as np
import pandas as pd
import nnConf
import nn
import nnUtils


#   Parse data and get the useful features.
def getDataValidation():
    path = os.path.dirname(__file__)
    trainUrl = os.path.join(path, "titanic", "train.csv")

    usefulFeatures = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
    outFeature = ["Survived"]
    # Returns a Python Pandas DataFrame
    inputData = pd.read_csv(trainUrl, usecols=usefulFeatures)
    outputData = pd.read_csv(trainUrl, usecols=outFeature)

    # Preprocessing training Data(Handle NaN)
    inputData["Cabin"] = inputData["Cabin"].map(lambda x: 1 if x == x else 0)
    inputData["Age"] = inputData["Age"].map(lambda x: x if x == x else 0)
    inputData["Sex"] = inputData["Sex"].map(lambda x: 1 if x == "male" else 0)

    # Replace all 'NaN' with Zeros
    for feat in inputData.columns:
        inputData[feat] = inputData[feat].map(lambda x: x if x == x else 0)

    # Normalize Data
    inputData = nnUtils.normalizeData(inputData, "Age")
    inputData = nnUtils.normalizeData(inputData, "Fare")
    inputData = nnUtils.normalizeData(inputData, "Pclass")
    inputData = nnUtils.normalizeData(inputData, "Parch")

#    print(pd.concat([inputData, outputData], axis=1))

    X = np.array(inputData)
    Y = np.array(outputData)

    return (X, Y)


def getTestData():
    path = os.path.dirname(__file__)
    testUrl = os.path.join(path, "titanic", "test.csv")

    usefulFeatures = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
    # Returns a Python Pandas DataFrame
    inputTestData = pd.read_csv(testUrl, usecols=usefulFeatures)
    PassengerId = pd.read_csv(testUrl, usecols=["PassengerId"])

    # Preprocessing test Data
    inputTestData["Cabin"] = inputTestData["Cabin"].map(lambda x: 1 if x == x else 0)
    inputTestData["Age"] = inputTestData["Age"].map(lambda x: x if x == x else 0)
    inputTestData["Sex"] = inputTestData["Sex"].map(lambda x: 1 if x == "male" else 0)

    for feat in inputTestData.columns:
        inputTestData[feat] = inputTestData[feat].map(lambda x: x if x == x else 0)

    # Normalize test Data
    inputTestData = nnUtils.normalizeData(inputTestData, "Age")
    inputTestData = nnUtils.normalizeData(inputTestData, "Fare")
    inputTestData = nnUtils.normalizeData(inputTestData, "Pclass")
    inputTestData = nnUtils.normalizeData(inputTestData, "Parch")

    testX = np.array(inputTestData)

    return (testX, PassengerId)


#     Get the best initial network configuration
#       * Decide and fix network depth(getNetConf).
def getInitialNetConf():
    # Network
    inputSize = 7
    layerCount = 2
    netConf = nnConf.NeuralNetworkConf(inputSize, layerCount)
    netConf.layerConf[0].neuronCount = 20
    netConf.layerConf[0].activationFn = "relu"
    netConf.layerConf[0].weightInitializerMethod = "random"
    netConf.layerConf[1].neuronCount = 1
    netConf.layerConf[1].activationFn = "sigmoid"
    netConf.layerConf[1].weightInitializerMethod = "random"
    netConf.Lambda = .00009      # 0.0001
    netConf.maxIter = 500

    return netConf


# Get the best value of Lambda
def getBestMaxIter(X, Y, netConf):
    bestMaxIter = 0
    bestAccuracy = 0
    for MaxIter in range(100, 1000, 100):
        netConf.maxIter = MaxIter
        n_splits = 5
        accuracyList = nnUtils.validateNN(X, Y, netConf, n_splits,
                                          showLearning=False,
                                          wtReuse=True,
                                          wtAndBias=nn.getGlobalConf())
        if np.mean(accuracyList) > bestAccuracy:
            bestAccuracy = np.mean(accuracyList)
            bestMaxIter = MaxIter

    return (bestMaxIter, bestAccuracy)


# Get the best value of Lambda
def getBestLambda(X, Y, netConf):
    bestLambda = 0
    bestAccuracy = 0
    for Lambda in np.linspace(0.00001, 0.0001, 200):
        netConf.Lambda = Lambda
        n_splits = 5
        accuracyList = nnUtils.validateNN(X, Y, netConf, n_splits,
                                          showLearning=False,
                                          wtReuse=True,
                                          wtAndBias=nn.getGlobalConf())
        if np.mean(accuracyList) > bestAccuracy:
            bestAccuracy = np.mean(accuracyList)
            bestLambda = Lambda

    return (bestLambda, bestAccuracy)


# Get the best value of neuron Count
def getBestNeuronCount(X, Y, netConf):
    bestNeuronCount = 0
    bestAccuracy = 0

    for neuronCount in range(2, 30):
        netConf.layerConf[0].neuronCount = neuronCount
        n_splits = 5
        accuracyList = nnUtils.validateNN(X, Y, netConf, n_splits,
                                          showLearning=False,
                                          wtReuse=True,
                                          wtAndBias=nn.getGlobalConf())

        if np.mean(accuracyList) > bestAccuracy:
            bestAccuracy = np.mean(accuracyList)
            bestNeuronCount = neuronCount

    return (bestNeuronCount, bestAccuracy)


def findAndStoreBestInitialWeights(X, Y, netConf, noOfPass):
    bestAccuracy = 0
    n_splits = 5

    for i in range(noOfPass):
        accuracyList = nnUtils.validateNN(X, Y, netConf, n_splits,
                                          showLearning=False)

        if np.mean(accuracyList) > bestAccuracy:
            bestAccuracy = np.mean(accuracyList)
            # Store Neural network settings and state
            print("Got Better accuracy: ", bestAccuracy)
            netConf.accuracy = bestAccuracy
            nnUtils.saveConfig(netConf, nn.getGlobalConf(), "nn.json")


def trainAndPredict(trainX, trainY, testX, PassengerId, netConf, wrightsBias):
    nn.setGlobalConf(wrightsBias[0], wrightsBias[1])
    NN = nn.NeuralNetwork(netConf, usePrevWt=True)

    # Train network with new data:
    T = nn.trainer(NN)
    T.maxIter = netConf.maxIter
    T.train(trainX, trainY, None, None)
#    T.train_GD(trainX, trainY, testX, testY)

    print("Final Training cost: ", T.J[-1])
    print("Number of iterations: ", len(T.J))

    testYhat = NN.forward(testX)

    # Consider values above .5 as 1 and values less that .5 as 0
    DBFunc = np.vectorize(lambda x: 0 if x < 0.5 else 1)
    testYAns = DBFunc(testYhat)
#    print(np.concatenate((PassengerId, testYAns), axis=1))

    testOutput = pd.DataFrame({"PassengerId": np.array(PassengerId).ravel(),
                               "Survived": np.array(testYAns).ravel()})
#    print(testOutput)
    path = os.path.dirname(__file__)
    resultUrl = os.path.join(path, "titanic", "result.csv")

    testOutput.to_csv(resultUrl, index=False)
    pass


def loadAndValidate(trainX, trainY):
    netConf, wrightsBias = nnUtils.restoreConfig("nn.json")

    accuracyList = nnUtils.validateNN(trainX, trainY, netConf, 5,
                                      showLearning=False,
                                      wtReuse=True,
                                      wtAndBias=wrightsBias)

    print("Best accuracy on validation: ", np.mean(accuracyList))


if __name__ == "__main__":
    #   1. Parse data and get the useful features
    trainX, trainY = getDataValidation()

    #   2. Get the best initial network configuration.
    #       * Decide and fix network depth(getNetConf).
    netConf = getInitialNetConf()

    #       * Run evaluation once with randomized initial weight and store
    #         the weight(Evaluate and store function, takes a flag for
    #         random initialization).
    n_splits = 5
    accuracyList = nnUtils.validateNN(trainX, trainY, netConf, n_splits,
                                      showLearning=False)
    print("Initial Mean Accuracy: ", np.mean(accuracyList))

    #       * Run evaluation varying different network parameters with
    #         the fixed initial weights.
#    netConf.Lambda, accuracy = getBestLambda(trainX, trainY, netConf)
#    netConf.maxIter, accuracy = getBestMaxIter(trainX, trainY, netConf)
    netConf.Lambda = 0.0000946
    netConf.maxIter = 500
    print("Best Lambda: ", netConf.Lambda)
    print("Best MaxIter: ", netConf.maxIter)
    print("Best Config Mean Accuracy: ", np.mean(accuracyList))

    #   3. With the best network configuration get best set of Weights.
    #       * Run and store the best weights for the network configuration
    #         in above step.
    #   4. Goto step 3 until maximum accuracy is achieved.
#    findAndStoreBestInitialWeights(trainX, trainY, netConf, 100)

    # 4.5. Validate with the best config
    loadAndValidate(trainX, trainY)

    #   5. Train the best model with training data and predict on test data.
    testX, PassengerId = getTestData()

    netConf, wrightsBias = nnUtils.restoreConfig("nn.json")

    trainAndPredict(trainX, trainY, testX, PassengerId, netConf, wrightsBias)

# That's all
