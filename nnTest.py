import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import nnConf
import nn


def testForOtherData(NN, X, Y):
    # Test network for various combinations of sleep and study
    hoursSleep = np.linspace(0, 10, 100)
    hoursStudy = np.linspace(0, 5, 100)

    # Normalize the data(same way training data was normalized)
    hoursSleepNorm = hoursSleep/10
    hoursStudyNorm = hoursStudy/5

    # Create 2D versions of input for plotting
    a, b = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

    # Join into a single input Matrix
    allInputs = np.zeros((a.size, 2))
    allInputs[:, 0] = a.ravel()
    allInputs[:, 1] = b.ravel()

    allOutputs = NN.forward(allInputs)

    # Make Contour Plot
    yy = np.dot(hoursStudy.reshape(100, 1), np.ones((1, 100)))
    xx = np.dot(hoursSleep.reshape(100, 1), np.ones((1, 100))).T

    cs = plt.contour(xx, yy, 100*allOutputs.reshape(100, 100))
    plt.clabel(cs, inline=1, fontsize=10)
    plt.xlabel("Hours Sleep")
    plt.ylabel("Hours study")
#    plt.show()


    fig = plt.figure()
    ax = fig.gca(projection = "3d")

    surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100), cmap=plt.cm.jet)
    sct = ax.scatter(X[:, 0], X[:, 1], Y)
    ax.set_xlabel("Hours Sleep")
    ax.set_ylabel("Hours Study")
    ax.set_zlabel("Test Score")

    plt.show()





def overittingTest():
    # Data
    # X = (hours sleeping, hours studying), y = score on test
    # Training Data
    trainX_org = np.array(([3, 5], [5, 1], [10, 2], [6, 1.5]), dtype=float)
    trainY_org = np.array(([75], [83], [93], [70]), dtype=float)

    # Testing Data
    testX =  np.array(([4, 5.5], [4.5, 1], [9, 2.5], [6, 2]), dtype=float)
    testY =  np.array(([70], [89], [85], [75]), dtype=float)

    # Normalize
    trainX = trainX_org/np.amax(trainX_org, axis=0)
    trainY = trainY_org/100

    testX = testX/np.amax(testX, axis=0)
    testY = testY/100

    # Network
    inputSize = 2
    layerCount = 2
    networkConf = nnConf.NeuralNetworkConf(inputSize, layerCount)
    networkConf.weightInitializerMethod = "random"  # Options: random, he, xavier, zeros, ones
    networkConf.layerConf[0].neuronCount = 3
    networkConf.layerConf[0].activationFn = "sigmoid"
    networkConf.layerConf[1].neuronCount = 1
    networkConf.layerConf[1].activationFn = "sigmoid"
    #networkConf.Lambda = 0.0001


    NN = nn.NeuralNetwork(networkConf)
    numgrad = NN.computeNumericalGradient(trainX, trainY)
    print("****************************")
    grad = NN.computeGradient(trainX, trainY)
    print("\nnumGrad: ", numgrad)
    print("\ngrad: ", grad)

    # Quantize numgrad and grad comparison(This should be < 1e-8)
    modelCorrectness = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
    print("\nModel Correctness: ", modelCorrectness)

    # Train network with new data:
    T = nn.trainer(NN)
    T.maxIter = 1000
    T.batchSize = 1
    T.learningRate = .5
#    T.train(trainX, trainY, testX, testY)
    T.train_GD(trainX, trainY, testX, testY)

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
    #testForOtherData(NN, trainX_org, trainY_org)





if __name__ == "__main__":
#    trainerTest()
#    NNTest()
#    NNTestComb()
#    overittingObservations()
    overittingTest()


# That's all
