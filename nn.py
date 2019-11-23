import numpy as np
from scipy import optimize
import costFunction

# Globals
globalWt = []
globalBias = []


def getGlobalConf():
    return (globalWt, globalBias)


def setGlobalConf(wt, bias):
    global globalWt
    global globalBias
    globalWt = wt
    globalBias = bias


def resetGlobalConf():
    global globalWt
    global globalBias
    globalWt = []
    globalBias = []


# Logistic function
def sigmoid(z):
    # Apply sigmoid activation function
    return 1.0/(1.0 + np.exp(-z))


# Derivative of logistic function
def sigmoidPrime(z):
    # Apply sigmoid activation function
    return np.exp(-z)/((1+np.exp(-z))**2)


def tanh(z):
    return np.tanh(z)


def tanhPrime(z):
    return (1-np.square(z))


def relu(z):
    return np.maximum(0, z)


def reluPrime(z):
    return 1.0*(z > 0)


# List of all activation functions
activationFnList = {"sigmoid":  (sigmoid, sigmoidPrime),
                    "tanh":     (tanh,    tanhPrime),
                    "relu":     (relu,    reluPrime)}


def initWeight(input, numOfNeurons, initMethod):
    if initMethod == "random":
        W = np.random.randn(input, numOfNeurons)

    elif initMethod == "he":
        W = np.random.randn(input, numOfNeurons)*np.sqrt(2/input)

    elif initMethod == "xavier":
        W = np.random.randn(input, numOfNeurons)*np.sqrt(1/input)

    elif initMethod == "zeros":
        W = np.zeros((input, numOfNeurons))

    elif initMethod == "ones":
        W = np.ones((input, numOfNeurons))

    return W


class annLayer(object):
    def __init__(self, netConf, idx, usePrevWt=False):
        if idx == 0:
            self.inputSize = netConf.inputSize
        else:
            self.inputSize = netConf.layerConf[idx-1].neuronCount

        self.numOfNeurons = netConf.layerConf[idx].neuronCount
        self.X = 0  # Input to this layer

        # Options: random, he, xavier, zeros, ones
        if usePrevWt is False:
            self.W = initWeight(self.inputSize,
                                self.numOfNeurons,
                                netConf.layerConf[idx].weightInitializerMethod)
            self.b = np.zeros((1, self.numOfNeurons))
            globalWt.append(self.W)
            globalBias.append(self.b)
        else:
            self.W = globalWt[idx]
            self.b = globalBias[idx]

        self.z = 0  # Dimension(no of examples * no of neurons)
        self.a = 0
        self.delta = 0
        self.dJdW = 0
        self.dJdb = 0
        self.activationFn = activationFnList.get(netConf.layerConf[idx].activationFn)[0]
        self.activationPrimeFn = activationFnList.get(netConf.layerConf[idx].activationFn)[1]
        pass


class NeuralNetwork(object):
    #
    def __init__(self, netConf, usePrevWt=False):
        self.netConf = netConf

        # if netConf.costFunctionName == "quadratic":
        self.costFunction = costFunction.quadraticCost
        self.costFunctionPrime = costFunction.quadraticCostPrime

        if usePrevWt is False:
            resetGlobalConf()

        self.layers = []
        for idx in range(self.netConf.layerCount):
            self.layers.append(annLayer(netConf, idx, usePrevWt=usePrevWt))

    #
    def forward(self, X):
        # Propagate inputs through Network
        # For First Layer
        self.layers[0].X = X
        self.layers[0].z = np.dot(X, self.layers[0].W) + self.layers[0].b   # Neuron activity
        self.layers[0].a = self.layers[0].activationFn(self.layers[0].z)    # Neuron activity

        # For each layer
        for idx in range(1, self.netConf.layerCount):
            self.layers[idx].X = self.layers[idx-1].a
            self.layers[idx].z = np.dot(self.layers[idx].X, self.layers[idx].W) + self.layers[idx].b  # Synapse activity
            self.layers[idx].a = self.layers[idx].activationFn(self.layers[idx].z)  # Neuron activity(Output of the layer)

        return self.layers[idx].a

    '''# Squared error cost(Regularized)
    def costFunction(self, X, y):
        yHat = self.forward(X)

        sqSum = 0
        # Calculate Squared sum of weights
        for idx in range(self.netConf.layerCount):
            sqSum = sqSum + np.sum(self.layers[idx].W**2)

        cost = (0.5*np.sum((y-yHat)**2))/X.shape[0] + ((self.netConf.Lambda/2)*sqSum)
        return cost

    # Squared error cost derivative(Regularized)
    # Calculates and updates the dJdW and dJdb term for each layer
    def costFunctionPrime(self, X, y):
        sampleCount = X.shape[0]
        layerCount = self.netConf.layerCount
        # Compute derivative with respect to W1 and W2
        yHat = self.forward(X)

        # For Output Layer
        self.layers[layerCount-1].delta = np.multiply(-(y-yHat),
                                                      self.layers[layerCount-1].activationPrimeFn(self.layers[layerCount-1].z))    # Element wise multiplication
        # Add gradient of regularization terms:
        self.layers[layerCount-1].dJdW = np.dot(self.layers[layerCount-1].X.T,
                                                self.layers[layerCount-1].delta)/sampleCount + \
                                         self.netConf.Lambda*self.layers[layerCount-1].W   # Matrix Multiplication

        bInput = np.ones((sampleCount, 1))
        self.layers[layerCount-1].dJdb = np.dot(bInput.T, self.layers[layerCount-1].delta)/sampleCount + \
                                         self.netConf.Lambda*self.layers[layerCount-1].b   # Matrix Multiplication

        for idx in reversed(range(layerCount-1)):
            self.layers[idx].delta = np.multiply(np.dot(self.layers[idx+1].delta, self.layers[idx+1].W.T),
                                                 self.layers[idx].activationPrimeFn(self.layers[idx].z))    # Element wise multiplication
            # Add gradient of regularization terms:
            self.layers[idx].dJdW = np.dot(self.layers[idx].X.T,
                                           self.layers[idx].delta)/sampleCount + \
                                           self.netConf.Lambda*self.layers[idx].W   # Matrix Multiplication

            bInput = np.ones((sampleCount, 1))
            self.layers[idx].dJdb = np.dot(bInput.T, self.layers[idx].delta)/sampleCount + \
                                             self.netConf.Lambda*self.layers[idx].b   # Matrix Multiplication
    '''

    # Helper functions

    def getParams(self):
        # Get all W and b rolled into vector
        params = []
        for layer in self.layers:
            params = np.concatenate((params, layer.W.ravel()))
            params = np.concatenate((params, layer.b.ravel()))

        return params

    def setParams(self, params):
        # Set all W and b using single parameter vector
        W_start = 0
        W_end = 0
        b_start = 0
        b_end = 0

        for layer in self.layers:
            W_end = W_start + (layer.inputSize * layer.numOfNeurons)
            layer.W = np.reshape(params[W_start:W_end], (layer.inputSize, layer.numOfNeurons))
            b_start = W_end
            b_end = b_start + layer.numOfNeurons
            layer.b = np.reshape(params[b_start:b_end], (1, layer.numOfNeurons))
            W_start = b_end

    def computeGradient(self, X, y):
        self.costFunctionPrime(self, X, y)

        params = []
        for layer in self.layers:
            params = np.concatenate((params, layer.dJdW.ravel()))
            params = np.concatenate((params, layer.dJdb.ravel()))

        return params

    def computeNumericalGradient(self, X, y):
        paramsInitial = self.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            perturb[p] = e
            self.setParams(paramsInitial + perturb)
            loss2 = self.costFunction(self, X, y)

            self.setParams(paramsInitial - perturb)
            loss1 = self.costFunction(self, X, y)

            # Compute Numerical Gradient
            numgrad[p] = (loss2-loss1) / (2*e)

            # Return the value we changed back to zero:
            perturb[p] = 0

        # Return parameters to original value:
        self.setParams(paramsInitial)

        return numgrad

###############################################################################


class trainer(object):
    def __init__(self, N):
        # Make local reference to Neural Network:
        self.N = N
        self.maxIter = 200
        self.batchSize = 1
        self.learningRate = .5

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(self.N, X, y)
        grad = self.N.computeGradient(X, y)

        return cost, grad

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.N, self.X, self.y))

        if self.testX is not None:
            self.testJ.append(self.N.costFunction(self.N, self.testX, self.testY))

    def train(self, trainX, trainY, testX, testY):
        # Make internal variable for callback function:
        self.X = trainX
        self.y = trainY

        self.testX = testX
        self.testY = testY

        # Make empty list to store cost:
        self.J = []
        self.testJ = []

        params0 = self.N.getParams()

        options = {"maxiter": self.maxIter, "disp": True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True,
                                 method="BFGS", args=(trainX, trainY),
                                 options=options, callback=self.callbackF)

        # Update trained weight:
        self.N.setParams(_res.x)
        self.optimizationResults = _res

    def train_GD(self, trainX, trainY, testX, testY):
        # Make empty list to store cost:
        self.J = []
        self.testJ = []
        batchSize = self.batchSize
        numOfBatch = int(np.ceil(np.shape(trainX)[0]/batchSize))
        batchStart = 0
        batchEnd = 0
        '''print("Num of Test case:", np.shape(trainX)[0])
        print("numOfBatch", numOfBatch)'''

        for i in range(self.maxIter):
            for batchIdx in range(numOfBatch):
                batchStart = batchIdx*batchSize
                batchEnd = batchStart + (batchSize if batchIdx < (numOfBatch-1) else (np.shape(trainX)[0]-batchStart))
                '''print("batchIdx", batchIdx)
                print("batchStart", batchStart)
                print("batchEnd", batchEnd)'''
                self.N.costFunctionPrime(self.N, trainX[batchStart:batchEnd], trainY[batchStart:batchEnd])
                for layer in self.N.layers:
                    layer.W = layer.W - self.learningRate*layer.dJdW
                    layer.b = layer.b - self.learningRate*layer.dJdb

#            self.N.netConf.learningRate = self.N.netConf.learningRate - ((5/100)*self.N.netConf.learningRate)    # Decaying learning rate

            self.J.append(self.N.costFunction(self.N, trainX, trainY))
            if testX.all() is not None:
                self.testJ.append(self.N.costFunction(self.N, testX, testY))


# That's all
