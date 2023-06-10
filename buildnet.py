import random

import numpy as np

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 0.3
REPLICATION_RATE = 0.1
CROSSOVER_RATE = 1 - REPLICATION_RATE
TOURNAMENT_SIZE = 15

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

    def __lt__(self, other):
        return self


# Genetic Algorithm
def genetic_algorithm():
    population = np.array([(0, NeuralNetwork()) for i in range(POPULATION_SIZE)],
                          dtype=[('score', float), ('net', NeuralNetwork)])

    for generation in range(MAX_GENERATIONS):
        print(f"Generation: {generation + 1}")
        sorted_population_scores = evaluate_and_sort(population)
        print("outside:")
        print(sorted_population_scores['score'])

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
        tournament_scores = random.choices(sorted_population, weights=sorted_population['score'], k=TOURNAMENT_SIZE)
        tournament_scores = np.sort(tournament_scores, order='score')
        tournament_winners[i] = tournament_scores[-1]

    for i in range(cross_size):
        parent1 = random.choice(tournament_winners)  # Extract the network object from the tuple
        parent2 = random.choice(tournament_winners)  # Extract the network object from the tuple

        child = NeuralNetwork()
        child.weights1 = np.copy(parent1['net'].weights1)
        child.weights2 = np.copy(parent2['net'].weights2)
        offsprings[i]['score'] = 0
        offsprings[i]['net'] = child

    return offsprings


def mutate(networks):
    for network in networks['net']:
        if np.random.random() < MUTATION_RATE:
            network.weights1 += np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.1
            network.weights2 += np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.1


# Run the genetic algorithm to train the network
best_network = genetic_algorithm()


# Load the test data
test_data = np.loadtxt("nn1.txt")
test_inputs = test_data[:, :-1]

# Load the trained network weights
best_network = NeuralNetwork()
weights = np.loadtxt("wnet.txt")
best_network.weights1 = weights[:INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
best_network.weights2 = weights[INPUT_SIZE * HIDDEN_SIZE:].reshape(HIDDEN_SIZE, OUTPUT_SIZE)

# Run the network on the test data
predictions = best_network.forward(X_test)

# zip the X_test and the predictions together and write them to a file
with open('predictions.txt', 'w') as f:
    for i in range(len(predictions)):
        f.write(f"prediction: {predictions[i]}\tsample: {X_test[i]}\tlabel: {y_test[i]}\n")
