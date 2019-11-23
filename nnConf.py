

class NeuralNetworkLayerConf(object):
    def __init__(self):
        self.neuronCount = 1
        self.activationFn = "sigmoid"
        self.weightInitializerMethod = "random"   # Options: random, he, xavier, zeros, ones
        pass


class NeuralNetworkConf(object):
    def __init__(self, inputSize, layerCount):
        self.inputSize = inputSize
        self.layerCount = layerCount

        # Initialize layer configurations
        self.layerConf = []
        for idx in range(self.layerCount):
            self.layerConf.append(NeuralNetworkLayerConf())

        self.Lambda = 0.0001
        self.maxIter = 1000
        self.learningRate = 3
        self.batchSize = 2
        self.accuracy = 0
        self.enableBias = True
        self.costFunctionName = "quadratic"

    def validateConf(self):
        # TBD
        pass
