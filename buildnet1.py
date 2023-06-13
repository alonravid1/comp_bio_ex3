import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
rng = np.random.default_rng()

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 50
MUTATION_RATE = 0.1
REPLICATION_RATE = 0.1
CROSSOVER_RATE = 1 - REPLICATION_RATE
TOURNAMENT_SIZE = 12
LEARNING_RATE = 0.1

sizes = [16, 32, 16, 1]

# TODO:
# 1 Build the network with the given parameters

# Load the training data - we need to make sure the loaded data is from nn1,
# depends on how it should work
# training_data = open("training_set.txt", "r").readlines()
# test_data = open("test_set.txt", "r").readlines()

# shuffle the data
rng.shuffle(training_data)
rng.shuffle(test_data)

# and split it by "\t" the data is before the '\t' and the label is after the '\t'

X_train = []
y_train = []

X_test = []
y_test = []

for i in range(len(training_data)):
    training_data[i] = training_data[i].split('   ')
    string = list(training_data[i][0])
    num_string = [int(i) for i in string]

    X_train.append(np.array(num_string))
    y_train.append(training_data[i][1].strip('\n'))

for i in range(len(test_data)):
    test_data[i] = test_data[i].split('   ')
    string = list(training_data[i][0])
    num_string = [int(i) for i in string]

    X_test.append(np.array(num_string))
    y_test.append(test_data[i][1].strip('\n'))


# Run the genetic algorithm to train the network
ga = GeneticAlgorithm(X_train, y_train, rng, sizes,
                       POPULATION_SIZE, MAX_GENERATIONS, REPLICATION_RATE, CROSSOVER_RATE,
                         MUTATION_RATE, LEARNING_RATE, TOURNAMENT_SIZE)
best_network = ga.run()
# best_network = NeuralNetwork()
# print(len(best_network.weights1))
# print(best_network.weights1.shape)

# Save the best network to a file
with open('wnet1.txt', 'w') as f:
    for weight in best_network.weights:
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]-1):
                f.write(f"{weight[i][j]},")
            f.write(f"{weight[i][-1]}")
            f.write("\n")
        f.write("end of layer\n")