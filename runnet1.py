import numpy as np
from NeuralNetwork import NeuralNetwork

validation_data = open("validation_set1.txt", "r").readlines()

# and split it by "\t" the data is before the '\t' and the label is after the '\t'

X_validation = []
y_validation = []

for i in range(len(validation_data)):
    validation_data[i] = validation_data[i].split('   ')
    string = list(validation_data[i][0])
    num_string = [int(i) for i in string]

    X_validation.append(np.array(num_string))
    y_validation.append(validation_data[i][1].strip('\n'))

# load a network from the wnet.txt file as above
best_network = NeuralNetwork()
with open('wnet1.txt', 'r') as f:
    weights_raw = f.readlines()
    
loaded_weights = [[]]
for line in weights_raw:
    if line.strip() == "end of layer":
        loaded_weights.append([])
    else:
        loaded_weights[-1].append([float(i) for i in line.split(',')])

best_network.weights = loaded_weights

# Run the network on the test data
predictions = best_network.forward(X_validation)

# zip the X_test and the predictions together and write them to a file
with open('predictions1.txt', 'w') as f:
    for prediction in predictions:
        f.write(prediction)