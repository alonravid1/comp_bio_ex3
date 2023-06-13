import random

import numpy as np

rng = np.random.default_rng()

# Genetic Algorithm Parameters
POPULATION_SIZE = 150
MAX_GENERATIONS = 100
MUTATION_RATE = 0.4
REPLICATION_RATE = 0.1
CROSSOVER_RATE = 1 - REPLICATION_RATE
TOURNAMENT_SIZE = 12

# Neural Network Parameters
INPUT_SIZE = 16
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 1
LEARNING_RATE = 0.1

sizes = [16, 32, 16, 1]

# perhaps replace with learning rate?
MUTATION_WEIGHT_MULTIPLYER = 0.1

# TODO:
# 1 Build the network with the given parameters

# Load the training data
data = open("nn0.txt", "r").readlines()
# shuffle the data
rng.shuffle(data)
# Split the data into training and test sets - 70% training and 30% testing
split_size = int(0.7 * len(data))

# and split it by "\t" the data is before the '\t' and the label is after the '\t'

x_data = []
y_data = []

for i in range(len(data)):
    data[i] = data[i].split('   ')
    string = list(data[i][0])
    num_string = [int(i) for i in string]

    x_data.append(np.array(num_string))
    y_data.append(data[i][1].strip('\n'))

X_train, X_test = x_data[:split_size], x_data[split_size:]
y_train, y_test = y_data[:split_size], y_data[split_size:]


# Define the neural network class
class NeuralNetwork:
    def __init__(self):
        self.weights1 = rng.standard_normal(size=(INPUT_SIZE, HIDDEN_SIZE_1))
        self.weights2 = rng.standard_normal(size=(HIDDEN_SIZE_1, HIDDEN_SIZE_2))
        self.weights3 = rng.standard_normal(size=(HIDDEN_SIZE_2, OUTPUT_SIZE))
        
        self.weights = []
        for i in range(len(sizes)-1):
            self.weights.append(rng.standard_normal(size=(sizes[i], sizes[i+1])))
            
    def forward(self, inputs):
        predictions = []
        for x in inputs:
            # original:
            # self.hidden_1 = np.dot(x, self.weights1)
            # self.hidden_activation_1 = self.relu(self.hidden_1)
            # self.hidden_2 = np.dot(self.hidden_activation_1, self.weights2)
            # self.hidden_activation_2 = self.relu(self.hidden_2)
            # self.output = np.dot(self.hidden_activation_2, self.weights3)
            # self.output_activation = self.relu(self.output)
            
            temp_x = np.copy(x)
            for weight in self.weights:
                self.hidden = np.dot(temp_x, weight)
                self.hidden_activation = self.relu(self.hidden)
                temp_x = self.hidden_activation
            
            # original: answer = x.flatten()
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


# Genetic Algorithm
def genetic_algorithm():
    population = np.array([(0, NeuralNetwork()) for i in range(POPULATION_SIZE)],
                          dtype=[('score', float), ('net', NeuralNetwork)])

    for generation in range(MAX_GENERATIONS):
        print(f"Generation: {generation + 1}")
        sorted_population_scores = evaluate_and_sort(population)
        # print the best network in the generation
        print(f"Best network: {sorted_population_scores[0]['net']}")
        # print the best score in the generation
        print(f"Best score: {sorted_population_scores[0]['score']}")

        elite_population = select(sorted_population_scores)
        crossover_population = crossover(sorted_population_scores)
        population = np.concatenate((elite_population, crossover_population))
        mutate(population)

    best_network = population[0]['net']
    return best_network


def evaluate_and_sort(population):
    for i in range(POPULATION_SIZE):
        predictions = population[i]['net'].forward(X_train)
        count = sum(predictions[i] == int(y_train[i]) for i in range(len(predictions)))
        accuracy = count / len(predictions)
        population[i]['score'] = accuracy
    population = np.sort(population, order='score')
    print("before:")
    print(population['score'])
    population = population[::-1]
    print("after:")
    print(population['score'])
    return population


def select(sorted_population):
    elite_count = int(REPLICATION_RATE * POPULATION_SIZE)
    return sorted_population[:elite_count]


def crossover(sorted_population):
    cross_size = int(POPULATION_SIZE * CROSSOVER_RATE)
    offsprings = np.array([(0, None) for i in range(cross_size)], dtype=[('score', float), ('net', NeuralNetwork)])
    tournament_winners = np.array([(0, None) for i in range(cross_size)],
                                  dtype=[('score', float), ('net', NeuralNetwork)])

    total_score = sorted_population['score'].sum()
    sorted_population['score'] = sorted_population['score'] / total_score
    for i in range(cross_size):
        tournament_scores = rng.choice(sorted_population, p=sorted_population['score'], size=TOURNAMENT_SIZE)
        tournament_scores = np.sort(tournament_scores, order='score')
        tournament_winners[i] = tournament_scores[-1]

    for i in range(cross_size):
        parent1 = rng.choice(tournament_winners)  # Extract the network object from the tuple
        parent2 = rng.choice(tournament_winners)  # Extract the network object from the tuple

        child = NeuralNetwork()
        random_assignments = rng.integers(0, 2, size=(len(sizes)-1))
        
        for i in range(len(random_assignments)):
            if random_assignments[i] == 0:
                child.weights[i] = np.copy(parent1['net'].weights[i])
            else:
                child.weights[i] = np.copy(parent2['net'].weights[i])
        
        # copy each child's weight from a parent at random
        # if random_assignments[0] == 0:
        #     child.weights1 = np.copy(parent1['net'].weights1)
        # else:
        #     child.weights1 = np.copy(parent2['net'].weights1)
        
        # if random_assignments[1] == 0:
        #     child.weights2 = np.copy(parent1['net'].weights2)
        # else:
        #     child.weights2 = np.copy(parent2['net'].weights2)
        
        # if random_assignments[2] == 0:
        #     child.weights3 = np.copy(parent1['net'].weights3)
        # else:
        #     child.weights3 = np.copy(parent2['net'].weights3)
                
        offsprings[i]['score'] = 0
        offsprings[i]['net'] = child

    return offsprings


def mutate(networks):
    for network in networks['net']:
        if rng.random() < MUTATION_RATE:
            for i in range(len(network.weights)):
                network.weights[i] += rng.standard_normal(size=network.weights[i].shape) * MUTATION_WEIGHT_MULTIPLYER
            # network.weights1 += rng.standard_normal(size=(INPUT_SIZE, HIDDEN_SIZE_1)) * MUTATION_WEIGHT_MULTIPLYER
            # network.weights2 += rng.standard_normal(size=(HIDDEN_SIZE_1, HIDDEN_SIZE_2)) * MUTATION_WEIGHT_MULTIPLYER
            # network.weights3 += rng.standard_normal(size=(HIDDEN_SIZE_2, OUTPUT_SIZE)) * MUTATION_WEIGHT_MULTIPLYER


# Run the genetic algorithm to train the network
best_network = genetic_algorithm()
# best_network = NeuralNetwork()
# print(len(best_network.weights1))
# print(best_network.weights1.shape)

# Save the best network to a file
with open('wnet.txt', 'w') as f:
    for weight in best_network.weights:
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]-1):
                f.write(f"{weight[i][j]},")
            f.write(f"{weight[i][-1]}")
            f.write("\n")
        f.write("end of layer\n")
        
#     for i in range(best_network.weights2.shape[0]):
#         for j in range(best_network.weights2.shape[1]-1):
#             f.write(f"{best_network.weights2[i][j]},")
#         f.write(f"{best_network.weights2[i][-1]}")
#         f.write("\n")
#     f.write("end of array\n")
#     for i in range(best_network.weights3.shape[0]):
#         for j in range(best_network.weights3.shape[1]-1):
#             f.write(f"{best_network.weights3[i][j]},")
#         f.write(f"{best_network.weights3[i][-1]}")
#         f.write("\n")

# Load the test data
test_data = np.loadtxt("nn1.txt")
test_inputs = test_data[:, :-1]

# load a network from the wnet.txt file as above

best_network = NeuralNetwork()
with open('wnet.txt', 'r') as f:
    weights_raw = f.readlines()
    
loaded_weights = [[]]
for line in weights_raw:
    if line.strip() == "end of layer":
        loaded_weights.append([])
    else:
        loaded_weights[-1].append([float(i) for i in line.split(',')])
    
# best_network.weights1 = np.array(weights[0], dtype=np.float64)
# best_network.weights2 = np.array(weights[1], dtype=np.float64)
# best_network.weights3 = np.array(weights[2], dtype=np.float64)
best_network.weights = loaded_weights
# Run the network on the test data
predictions = best_network.forward(X_test)

# zip the X_test and the predictions together and write them to a file
with open('predictions.txt', 'w') as f:
    for i in range(len(predictions)):
        f.write(f"prediction: {predictions[i]}\tsample: {X_test[i]}\tlabel: {y_test[i]}\n")
