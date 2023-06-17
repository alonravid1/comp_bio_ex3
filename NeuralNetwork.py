import numpy as np

# Define the neural network class
class NeuralNetwork:
    def __init__(self, rng, sizes):
        self.weights = []
        self.rng = rng
        self.sizes = sizes
        
        for i in range(len(self.sizes)-1):
            self.weights.append(self.rng.standard_normal(size=(self.sizes[i], self.sizes[i+1])))
            
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
