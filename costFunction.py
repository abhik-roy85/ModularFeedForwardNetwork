import numpy as np


# Quadratic cost:
# Also known as mean squared error, maximum likelihood, and sum squared error.
# Squared error cost(Regularized)
def quadraticCost(NN, X, y):
    yHat = NN.forward(X)

    sqSum = 0
    # Calculate Squared sum of weights
    for idx in range(NN.netConf.layerCount):
        sqSum = sqSum + np.sum(NN.layers[idx].W**2)

    cost = (0.5*np.sum((y-yHat)**2))/X.shape[0] + ((NN.netConf.Lambda/2)*sqSum)
    return cost


# Squared error cost derivative(Regularized)
# Calculates and updates the dJdW and dJdb term for each layer
def quadraticCostPrime(NN, X, y):
    sampleCount = X.shape[0]
    layerCount = NN.netConf.layerCount
    # Compute derivative with respect to W1 and W2
    yHat = NN.forward(X)

    # For Output Layer
    NN.layers[layerCount-1].delta = np.multiply(-(y-yHat),
                                                  NN.layers[layerCount-1].activationPrimeFn(NN.layers[layerCount-1].z))    # Element wise multiplication

    # Add gradient of regularization terms:
    NN.layers[layerCount-1].dJdW = np.dot(NN.layers[layerCount-1].X.T, NN.layers[layerCount-1].delta)/sampleCount + \
                                          NN.netConf.Lambda*NN.layers[layerCount-1].W

    bInput = np.ones((sampleCount, 1))
    NN.layers[layerCount-1].dJdb = np.dot(bInput.T, NN.layers[layerCount-1].delta)/sampleCount + \
                                   NN.netConf.Lambda*NN.layers[layerCount-1].b

    # For all Hidden Layers
    for idx in reversed(range(layerCount-1)):
        NN.layers[idx].delta = np.multiply(np.dot(NN.layers[idx+1].delta, NN.layers[idx+1].W.T),
                                                  NN.layers[idx].activationPrimeFn(NN.layers[idx].z))    # Element wise multiplication

        # Add gradient of regularization terms:
        NN.layers[idx].dJdW = np.dot(NN.layers[idx].X.T, NN.layers[idx].delta)/sampleCount + \
                                     NN.netConf.Lambda * NN.layers[idx].W

        bInput = np.ones((sampleCount, 1))
        NN.layers[idx].dJdb = np.dot(bInput.T, NN.layers[idx].delta)/sampleCount + \
                                         NN.netConf.Lambda * NN.layers[idx].b


###############################################################################
# Cross-entropy cost:
# Also known as Bernoulli negative log-likelihood and Binary Cross-Entropy.
def CrossEntropy(yHat, y):
    if y == 1:
        return -np.log(yHat)
    else:
        return -np.log(1 - yHat)


# That's all
