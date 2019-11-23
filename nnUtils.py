import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import nn
import nnConf


def convertNdarrayToList(itemList):
    topList = []
    for item in itemList:
        if type(item) is not np.ndarray:
            if type(item) is not int:
                topList.append(convertNdarrayToList(item))
            else:
                topList.append(item.tolist())
        else:
            topList.append(item.tolist())

    return topList


def convertListToNdarray(itemList):
    topList = []
    for item in itemList:
        topList.append(np.array(item))

    return topList


def dfFltr(df, colName, value):
    return df.loc[lambda df: df[colName] == value]


def normalizeData(df, feat):
    # Normalize the data
    min = df.loc[:, feat].min()
    max = df.loc[:, feat].max()
    mean = df.loc[:, feat].mean()
    df.loc[:, feat] = df.loc[:, feat].map(lambda x: (x-mean)/(max-min))    # Mean Normalize Data

    # Make the data spread from zero to one.
    min = df.loc[:, feat].min()
    df.loc[:, feat] = df.loc[:, feat].map(lambda x: x-min)
    max = df.loc[:, feat].max()
    df.loc[:, feat] = df.loc[:, feat].map(lambda x: x/max)

    return df


def validateNN(dataX, dataY, netConf, n_splits,
               showLearning=True,
               wtReuse=False,
               wtAndBias=[]):

    print("\n*********************************************")
    accuracyList = []

    # Validate Network
    NN = nn.NeuralNetwork(netConf, usePrevWt=False)

    # Validate Network
    numgrad = NN.computeNumericalGradient(dataX, dataY)
    grad = NN.computeGradient(dataX, dataY)

    # Quantize numgrad and grad comparison(This should be < 1e-8)
    modelCorrectness = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
    print("\nModel Correctness: ", modelCorrectness)

    if wtReuse is True:
        nn.setGlobalConf(wtAndBias[0], wtAndBias[1])

    # break data into training and test set
    kf = KFold(n_splits=n_splits, random_state=None)
    for train_index, test_index in kf.split(dataX):
        # Split data into training and test set
        trainX, testX = dataX[train_index], dataX[test_index]
        trainY, testY = dataY[train_index], dataY[test_index]

        NN = nn.NeuralNetwork(netConf, usePrevWt=wtReuse)
        wtReuse = True

        '''# Validate Network
        numgrad = NN.computeNumericalGradient(trainX, trainY)
        grad = NN.computeGradient(trainX, trainY)

        # Quantize numgrad and grad comparison(This should be < 1e-8)
        modelCorrectness = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
        print("\nModel Correctness: ", modelCorrectness)'''

        # Train network with new data:
        T = nn.trainer(NN)
        T.maxIter = netConf.maxIter
        T.train(trainX, trainY, testX, testY)
    #    T.train_GD(trainX, trainY, testX, testY)

        # Show Learning for training and test dataset
        if showLearning is True:
            plt.plot(T.J)
            plt.plot(T.testJ)
            plt.grid(1)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.legend(["Training", "Testing"])
            plt.show()

        print("Final Training cost: ", T.J[-1])
        print("Final Test cost: ", T.testJ[-1])
        print("Number of iterations: ", len(T.J))

        testYhat = NN.forward(testX)

        # Consider values above .5 as 1 and values less that .5 as 0
        DBFunc = np.vectorize(lambda x: 0 if x<0.5 else 1)
        testYAns = DBFunc(testYhat)
    #    print(np.concatenate((testY, testYAns, testYhat, (testY == testYAns)), axis=1))

        accuracy = np.count_nonzero(testY == testYAns)/testY.shape[0]
        print("Accuracy: ", accuracy*100, "%")
        accuracyList.append(accuracy)
        print("*********************************************\n")

    return accuracyList


def saveConfig(netConf, wrightsBias, confFile):
    conf = {"inputSize": netConf.inputSize,
            "layerCount": netConf.layerCount,
            "Lambda": netConf.Lambda,
            "maxIter": netConf.maxIter,
            "learningRate": netConf.learningRate,
            "batchSize": netConf.batchSize,
            "accuracy": netConf.accuracy,
            "enableBias": netConf.enableBias,
            }

    # Store layer-wise configurations
    for i in range(netConf.layerCount):
        conf["W"+str(i)] = {"neuronCount": netConf.layerConf[i].neuronCount,
                            "activationFn": netConf.layerConf[i].activationFn,
                            "weightInitializerMethod": netConf.layerConf[i].weightInitializerMethod}

    # Convert all ndarrays' to list for json and store to dictionary
    wrightsBias = convertNdarrayToList(wrightsBias)
    conf["wrightsBias"] = {"weight": wrightsBias[0], "bias": wrightsBias[1]}

    # Save to json file
    path = os.path.dirname(__file__)
    confUrl = os.path.join(path, "titanic", confFile)
    with open(confUrl, "w") as fp:
        json.dump(conf, fp)


def restoreConfig(confFile):
    # Open ang load json file
    path = os.path.dirname(__file__)
    confUrl = os.path.join(path, "titanic", confFile)
    with open(confUrl, "r") as fp:
        conf = json.load(fp)

    # Create a nnConf object
    netConf = nnConf.NeuralNetworkConf(conf["inputSize"], conf["layerCount"])

    netConf.inputSize = conf["inputSize"]
    netConf.layerCount = conf["layerCount"]
    netConf.Lambda = conf["Lambda"]
    netConf.maxIter = conf["maxIter"]
    netConf.learningRate = conf["learningRate"]
    netConf.batchSize = conf["batchSize"]
    netConf.accuracy = conf["accuracy"]
    netConf.enableBias = conf["enableBias"]

    # Restore layer-wise configurations
    for i in range(netConf.layerCount):
        netConf.layerConf[i].neuronCount = conf["W"+str(i)]["neuronCount"]
        netConf.layerConf[i].activationFn = conf["W"+str(i)]["activationFn"]
        netConf.layerConf[i].weightInitializerMethod = conf["W"+str(i)]["weightInitializerMethod"]

    # Restore the Weights and biases
    wrightsBias = (convertListToNdarray(conf["wrightsBias"]["weight"]),
                   convertListToNdarray(conf["wrightsBias"]["bias"]))

#    print("While Loading", wrightsBias)

    return (netConf, wrightsBias)


# That's all
