import numpy as np


def xavier_init(shape):
    '''Xavier initialization for the weights'''
    n_inputs, n_outputs = shape[1], shape[0]
    limit = np.sqrt(6 / (n_inputs + n_outputs))
    return np.random.uniform(-limit, limit, shape)


# Define the neural network class
class NeuralNetwork:
    def __init__(self, rng, sizes):
        self.weights = []
        self.rng = rng
        self.sizes = sizes

        if self.weights == []:
            for i in range(len(self.sizes) - 1):
                weight_shape = (self.sizes[i], self.sizes[i + 1])
                weight_init = xavier_init(weight_shape)
                self.weights.append(weight_init)

    def forward(self, inputs):
        x = np.copy(inputs)
        for weight in self.weights:
            self.hidden = np.dot(x, weight)
            self.hidden_activation = self.sigmoid(self.hidden)
            x = self.hidden_activation

        answer = x.flatten()
        predictions = np.where(answer > 0.5, 1, 0)

        return predictions

    # activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def __lt__(self, other):
        return self
