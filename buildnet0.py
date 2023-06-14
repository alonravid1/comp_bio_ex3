import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
rng = np.random.default_rng()

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 50
MUTATION_RATE = 0.4
REPLICATION_RATE = 0.1
CROSSOVER_RATE = 1 - REPLICATION_RATE
TOURNAMENT_SIZE = 45
LEARNING_RATE = 0.05

sizes = [16, 32, 16, 1]

# TODO:
# 1 Build the network with the given parameters

# Load the training data
training_data = open("training_set.txt", "r").readlines()
test_data = open("test_set.txt", "r").readlines()

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
# ga = GeneticAlgorithm(X_train, y_train, rng, sizes,
#                        POPULATION_SIZE, MAX_GENERATIONS, REPLICATION_RATE, CROSSOVER_RATE,
#                          MUTATION_RATE, LEARNING_RATE, TOURNAMENT_SIZE)
# best_network = ga.run()


with open('results0.csv', 'w') as res_file:
    res_file.write("POPULATION_SIZE,MAX_GENERATIONS,REPLICATION_RATE,CROSSOVER_RATE,MUTATION_RATE,LEARNING_RATE,TOURNAMENT_SIZE\n")

params = [0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5]

# run genetic algorithm to train the network over several parameters
for param in params:
    ga = GeneticAlgorithm(X_train, y_train, rng, sizes,
                       POPULATION_SIZE, MAX_GENERATIONS, REPLICATION_RATE, CROSSOVER_RATE,
                         MUTATION_RATE, param, TOURNAMENT_SIZE)
    best_network = ga.run()

    with open('results0.csv', 'a') as res_file:
        res_file.write(f"{sizes},{POPULATION_SIZE},{MAX_GENERATIONS},{REPLICATION_RATE},{CROSSOVER_RATE},{MUTATION_RATE},{param},{TOURNAMENT_SIZE}\n")
        predictions = best_network.forward(X_train)
        count = sum(predictions[i] == int(y_train[i]) for i in range(len(predictions)))
        accuracy = count / len(predictions)
        res_file.write(f"best score: {accuracy}\n")

# Save the best network to a file
with open('wnet0.txt', 'w') as f:
    for weight in best_network.weights:
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]-1):
                f.write(f"{weight[i][j]},")
            f.write(f"{weight[i][-1]}")
            f.write("\n")
        f.write("end of layer\n")