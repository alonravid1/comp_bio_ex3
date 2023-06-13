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
        predictions = []
        for x in inputs:           
            temp_x = np.copy(x)
            for weight in self.weights:
                self.hidden = np.dot(temp_x, weight)
                self.hidden_activation = self.sigmoid(self.hidden)
                temp_x = self.hidden_activation
            
            answer = temp_x.flatten()
            if answer >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    # activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def __lt__(self, other):
        return self
