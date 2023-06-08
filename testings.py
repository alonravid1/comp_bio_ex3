import numpy as np

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 0.01

# Neural Network Parameters
INPUT_SIZE = 16
HIDDEN_SIZE = 32
OUTPUT_SIZE = 1
LEARNING_RATE = 0.1

# Load the training data
train_data = np.loadtxt("nn0.txt")
train_inputs = train_data[:, :-1]
train_labels = train_data[:, -1]

# Normalize the inputs
train_inputs = train_inputs / np.max(train_inputs, axis=0)


# Define the neural network class
class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE)
        self.weights2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, inputs):
        self.hidden = np.dot(inputs.T, self.weights1)
        self.hidden_activation = self.sigmoid(self.hidden)
        self.output = np.dot(self.hidden_activation, self.weights2)
        self.output_activation = self.sigmoid(self.output)
        return self.output_activation.flatten()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Genetic Algorithm
def genetic_algorithm():
    population = []
    for _ in range(POPULATION_SIZE):
        network = NeuralNetwork()
        population.append(network)

    for generation in range(MAX_GENERATIONS):
        print(f"Generation: {generation + 1}")
        scores = []
        for network in population:
            scores.append(evaluate(network))

        population = select(population, scores)
        population = crossover(population)
        mutate(population)

    best_network = population[0]
    return best_network


def evaluate(network):
    predictions = network.forward(train_inputs)
    accuracy = np.mean((predictions >= 0.5) == train_labels.reshape(-1, 1))
    return accuracy


def select(population, scores):
    sorted_population = [x for _, x in sorted(zip(scores, population), reverse=True)]
    elite_count = int(0.2 * POPULATION_SIZE)
    return sorted_population[:elite_count]


def crossover(population):
    offspring = []
    elite_count = len(population)
    offspring.extend(population[:elite_count])

    while len(offspring) < POPULATION_SIZE:
        parent1 = np.random.choice(population)
        parent2 = np.random.choice(population)
        child = NeuralNetwork()
        child.weights1 = np.copy(parent1.weights1)
        child.weights2 = np.copy(parent2.weights2)
        offspring.append(child)

    return offspring


def mutate(population):
    for network in population:
        if np.random.random() < MUTATION_RATE:
            network.weights1 += np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.1
            network.weights2 += np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.1


# Load the two datasets
nn0_data = np.loadtxt("nn0.txt")
nn1_data = np.loadtxt("nn1.txt")

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
