import numpy as np
from NeuralNetwork import NeuralNetwork


if __name__ == "__main__":
    
    print("starting\nrunning...")
    
    validation_data = open("testnet0.txt", "r").readlines()

    X_validation = []
    y_validation = []

    for i in range(len(validation_data)):
        validation_data[i] = validation_data[i].split('   ')
        if len(validation_data[i]) == 1:
            string = list(validation_data[i][0])
            num_string = [int(j) for j in string]

            X_validation.append(np.array(num_string))
        else:
            string = list(validation_data[i][0])
            num_string = [int(j) for j in string]

            X_validation.append(np.array(num_string))
            y_validation.append(validation_data[i][1].strip('\n'))

    # load a network from the wnet.txt file as above
    best_network = NeuralNetwork()
    with open('wnet0.txt', 'r') as f:
        weights_raw = f.readlines()
        
    loaded_weights = [[]]
    for line in weights_raw:
        if line.strip() == "end of layer":
            loaded_weights.append([])
        else:
            loaded_weights[-1].append([float(i) for i in line.split(',')])
    loaded_weights.pop()


    best_network.weights = loaded_weights
    
    # Run the network on the test data
    predictions = best_network.forward(X_validation)

    # zip the X_test and the predictions together and write them to a file
    with open('predictions0.txt', 'w') as f:
        for i in range(len(predictions)-1):
            f.write(str(predictions[i]) + "\n")
        f.write(str(predictions[-1]))

    input("Done!\nEnter to exit")