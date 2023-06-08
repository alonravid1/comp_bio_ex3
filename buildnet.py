import random

import numpy as np

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 0.01
REPLICATION_RATE = 0.2
CROSSOVER_RATE = 1 - REPLICATION_RATE
TOURNAMENT_SIZE = 20

# Neural Network Parameters
INPUT_SIZE = 16
HIDDEN_SIZE = 32
OUTPUT_SIZE = 1
LEARNING_RATE = 0.1

# Load the training data
data = open("nn0.txt", "r").readlines()
print(data[0])
# Split the data into training and test sets
split_size = int(0.8 * len(data))

# i want to split the data into 80% training and 20% testing
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
        self.weights1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE)
        self.weights2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, inputs):
        predictions = []
        for x in inputs:
            self.hidden = np.dot(x, self.weights1)
            self.hidden_activation = self.sigmoid(self.hidden)
            self.output = np.dot(self.hidden_activation, self.weights2)
            self.output_activation = self.sigmoid(self.output)
            answer = self.output_activation.flatten()
            if answer >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Genetic Algorithm
def genetic_algorithm():
    net_population_scores = []
    score = 0
    for _ in range(POPULATION_SIZE):
        network = NeuralNetwork()
        net_population_scores.append((network, score))

    for generation in range(MAX_GENERATIONS):
        print(f"Generation: {generation + 1}")
        sorted_population_scores = evaluate_and_sort(net_population_scores)
        elite_population = select(sorted_population_scores)
        crossover_population = crossover(sorted_population_scores)
        net_population_scores = elite_population + crossover_population
        mutate(net_population_scores)

    best_network[0] = net_population_scores[0]
    return best_network


def evaluate_and_sort(population_scores_tuple):
    accuracies = []
    for network in population_scores_tuple:
        predictions = network[0].forward(X_train)
        count = sum(predictions[i] == int(y_train[i]) for i in range(len(predictions)))
        accuracy = count / len(predictions)
        accuracies.append(accuracy)
    sorted_population_scores = sorted(zip(population_scores_tuple, accuracies), key=lambda x: x[1], reverse=True)
    return sorted_population_scores


def select(sorted_population):
    elite_count = int(REPLICATION_RATE * POPULATION_SIZE)
    return sorted_population[:elite_count]


def crossover(sorted_population_score_tuple):
    offspring = []
    tournament_winners = []
    weights = [pair[1] for pair in sorted_population_score_tuple]
    total_weight = sum(weights)
    probability_dist = [w / total_weight for w in weights]

    for i in range(int(CROSSOVER_RATE * POPULATION_SIZE)):
        tournament_scores = random.choices(sorted_population_score_tuple, weights=probability_dist, k=TOURNAMENT_SIZE)
        tournament_scores.sort(key=lambda x: x[1], reverse=True)
        tournament_winners.append(tournament_scores[0])

    for i in range(int(CROSSOVER_RATE * POPULATION_SIZE)):
        parent1 = random.choice(tournament_winners)[0]  # Extract the network object from the tuple
        print(parent1)
        parent2 = random.choice(tournament_winners)[0]  # Extract the network object from the tuple
        print(parent2)

        child = NeuralNetwork()
        child.weights1 = np.copy(parent1.weights1)
        child.weights2 = np.copy(parent2.weights2)
        offspring.append(child)

    return offspring


def mutate(networks):
    for network in networks:
        if np.random.random() < MUTATION_RATE:
            network.weights1 += np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.1
            network.weights2 += np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.1


# Run the genetic algorithm to train the network
best_network = genetic_algorithm()

# Save the best network to a file
np.savetxt("wnet.txt", np.concatenate((best_network.weights1.flatten(), best_network.weights2.flatten())))

# Load the test data
test_data = np.loadtxt("nn1.txt")
test_inputs = test_data[:, :-1]

# Normalize the inputs
test_inputs = test_inputs / np.max(test_inputs, axis=0)

# Load the trained network weights
weights = np.loadtxt("wnet.txt")
best_network.weights1 = weights[:INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
best_network.weights2 = weights[INPUT_SIZE * HIDDEN_SIZE:].reshape(HIDDEN_SIZE, OUTPUT_SIZE)

# Run the network on the test data
predictions = best_network.forward(test_inputs)

# Save the predictions to a file
np.savetxt("predictions.txt", predictions)
