import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nnConf
import nn
import nnUtils


def getDataTest():
    path = os.path.dirname(__file__)
    trainUrl = os.path.join(path, "titanic", "train.csv")
    testUrl = os.path.join(path, "titanic", "test.csv")

    usefulFeatures = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
    outFeature = ["Survived"]
    inputData = pd.read_csv(trainUrl, usecols=usefulFeatures)     # Returns a Python Pandas DataFrame
    outputData = pd.read_csv(trainUrl, usecols=outFeature)     # Returns a Python Pandas DataFrame
    inputTestData = pd.read_csv(testUrl, usecols=usefulFeatures)     # Returns a Python Pandas DataFrame
    PassengerId = pd.read_csv(testUrl, usecols=["PassengerId"])

    # Preprocessing training Data
    inputData["Cabin"] = inputData["Cabin"].map(lambda x: 1 if x == x else 0)  # Handle NaN
    inputData["Age"] = inputData["Age"].map(lambda x: x if x == x else 0)  # Handle NaN
    inputData["Sex"] = inputData["Sex"].map(lambda x: 1 if x == "male" else 0)

    # Normalize Data
    inputData = nnUtils.normalizeData(inputData, "Age")
    inputData = nnUtils.normalizeData(inputData, "Fare")
    inputData = nnUtils.normalizeData(inputData, "Pclass")
    inputData = nnUtils.normalizeData(inputData, "Parch")

    # Preprocessing test Data
    inputTestData["Cabin"] = inputTestData["Cabin"].map(lambda x: 1 if x == x else 0)  # Handle NaN
    inputTestData["Age"] = inputTestData["Age"].map(lambda x: x if x == x else 0)  # Handle NaN
    inputTestData["Sex"] = inputTestData["Sex"].map(lambda x: 1 if x == "male" else 0)

    for feat in inputTestData.columns:
        inputTestData[feat] = inputTestData[feat].map(lambda x: x if x == x else 0)

    # Normalize test Data
    inputTestData = nnUtils.normalizeData(inputTestData, "Age")
    inputTestData = nnUtils.normalizeData(inputTestData, "Fare")
    inputTestData = nnUtils.normalizeData(inputTestData, "Pclass")
    inputTestData = nnUtils.normalizeData(inputTestData, "Parch")

#    print(pd.concat([inputData, outputData], axis=1))

    trainX = np.array(inputData)
    trainY = np.array(outputData)
    testX = np.array(inputTestData)

    return (trainX, trainY, testX, PassengerId)


def titanicTest():
    # Data
    trainX, trainY, testX, PassengerId = getDataTest()

    # Network
    inputSize = 7
    layerCount = 2
    networkConf = nnConf.NeuralNetworkConf(inputSize, layerCount)
    networkConf.layerConf[0].neuronCount = 20
    networkConf.layerConf[0].activationFn = "relu"
    networkConf.layerConf[0].weightInitializerMethod = "random"
    networkConf.layerConf[1].neuronCount = 1
    networkConf.layerConf[1].activationFn = "sigmoid"
    networkConf.layerConf[1].weightInitializerMethod = "random"
    networkConf.Lambda = 0.00009
    networkConf.maxIter = 500

    NN = nn.NeuralNetwork(networkConf)

    # Train network with new data:
    T = nn.trainer(NN)
    T.maxIter = networkConf.maxIter
    T.train(trainX, trainY, None, None)
#    T.train_GD(trainX, trainY, testX, testY)

    print("Final Training cost: ", T.J[-1])
    print("Number of iterations: ", len(T.J))

    testYhat = NN.forward(testX)

    # Consider values above .5 as 1 and values less that .5 as 0
    DBFunc = np.vectorize(lambda x: 0 if x < 0.5 else 1)
    testYAns = DBFunc(testYhat)
#    testYAns = np.int(testYAns)
#    print(np.shape(testYAns))
#    print(np.concatenate((PassengerId, testYAns), axis=1))

    testOutput = pd.DataFrame({"PassengerId": np.array(PassengerId).ravel(),
                               "Survived": np.array(testYAns).ravel()})
    print(testOutput)
    path = os.path.dirname(__file__)
    resultUrl = os.path.join(path, "titanic", "result.csv")

    testOutput.to_csv(resultUrl, index=False)

##############################################################################


def getDataValidation():
    path = os.path.dirname(__file__)
    trainUrl = os.path.join(path, "titanic", "train.csv")

    usefulFeatures = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
    outFeature = ["Survived"]
    inputData = pd.read_csv(trainUrl, usecols=usefulFeatures)     # Returns a Python Pandas DataFrame
    outputData = pd.read_csv(trainUrl, usecols=outFeature)     # Returns a Python Pandas DataFrame

    # Preprocessing training Data
    inputData["Cabin"] = inputData["Cabin"].map(lambda x: 1 if x == x else 0)  # Handle NaN
    inputData["Age"] = inputData["Age"].map(lambda x: x if x == x else 0)  # Handle NaN
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


def validateTitanic():
    # Data
    X, Y = getDataValidation()

    # Network
    inputSize = 7
    layerCount = 2
    networkConf = nnConf.NeuralNetworkConf(inputSize, layerCount)
    networkConf.layerConf[0].neuronCount = 20
    networkConf.layerConf[0].activationFn = "relu"
    networkConf.layerConf[0].weightInitializerMethod = "random"
    networkConf.layerConf[1].neuronCount = 1
    networkConf.layerConf[1].activationFn = "sigmoid"
    networkConf.layerConf[1].weightInitializerMethod = "random"
    networkConf.Lambda = .00009      # 0.0001
    networkConf.maxIter = 500

    accuracyList = nnUtils.validateNN(X, Y, networkConf, 5, showLearning=False)

    print(accuracyList)
    print("Mean Accuracy: ", np.mean(accuracyList))

    # Store Neural network settings and state
#    print(NN.getGlobalConf())
#    print(networkConf)
    nnUtils.saveConfig(networkConf, nn.getGlobalConf())

    nnUtils.restoreConfig()

    accuracyList = []
    return (accuracyList)


def testAccuracyFor():
    accuracy = []
    xAxis = []
#    for i in np.linspace(0.00001, 0.0001, 20):
    for i in np.linspace(50, 2000, 20):
        accuracy.append(np.mean(validateTitanic(i)))
        xAxis.append(i)

    plt.plot(xAxis, accuracy)
    plt.show()

    # Accuracy platues beyond 20 neurons in the hidden layer.


def runAndStore():
    # Data
    X, Y = getDataValidation()

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

    accuracyList = nnUtils.validateNN(X, Y, netConf, 5,
                                      showLearning=False)

    netConf.accuracy = np.mean(accuracyList)
    print(accuracyList)
    print("Mean Accuracy: ", netConf.accuracy)

    # Store Neural network settings and state
    nnUtils.saveConfig(netConf, nn.getGlobalConf(), "nn.json")


def loadAndRun():
    # Data
    X, Y = getDataValidation()

    netConf, wrightsBias = nnUtils.restoreConfig("nn.json")

    accuracyList = nnUtils.validateNN(X, Y, netConf, 5,
                                      showLearning=False,
                                      wtReuse=True,
                                      wtAndBias=wrightsBias)

    print(accuracyList)
    print("Mean Accuracy: ", np.mean(accuracyList))


def runAndMeasure():
    # Data
    X, Y = getDataValidation()

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

    accuracyList = nnUtils.validateNN(X, Y, netConf, 5,
                                      showLearning=False)

    netConf.accuracy = np.mean(accuracyList)
    print(accuracyList)
    print("Mean Accuracy: ", netConf.accuracy)

    return (netConf, nn.getGlobalConf())


def findBestInitialWeights():
    netConf, wrightsBias = runAndMeasure()

    netConfOld, wrightsBiasOld = nnUtils.restoreConfig("nn.json")

    print("New accuracy: ", netConf.accuracy)
    print("Old accuracy: ", netConfOld.accuracy)

    if netConf.accuracy > netConfOld.accuracy:
        # Store Neural network settings and state
        print("Got Better accuracy")
        nnUtils.saveConfig(netConf, wrightsBias, "nn.json")


if __name__ == "__main__":
#    titanicTest()
#    testAccuracyFor()
#    validateTitanic()
#    runAndStore()
    loadAndRun()

#    for i in range(100):
#        findBestInitialWeights()


# That's all
