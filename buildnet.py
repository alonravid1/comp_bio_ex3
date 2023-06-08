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


# print("X_train: ", X_train[0:5])
# print("X_test: ", X_test[0:5])
# print("y_train: ", y_train[0:5])
# print("y_test: ", y_test[0:5])

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
    net_population = []
    for _ in range(POPULATION_SIZE):
        network = NeuralNetwork()
        net_population.append(network)

    for generation in range(MAX_GENERATIONS):
        print(f"Generation: {generation + 1}")
        scores = evaluate(net_population)
        print(scores)
        net_population = select(net_population, scores)
        net_population = crossover(net_population)
        mutate(net_population)

    best_network = net_population[0]
    return best_network


def evaluate(population):
    scores = []
    for network in population:
        predictions = network.forward(X_train)
        count = 0
        for i in range(len(predictions)):
            if predictions[i] == y_train[i]:
                count += 1

        accuracy = count / len(predictions)
        scores.append(accuracy)
    return scores


def select(population, scores):
    sorted_population = [x for x, _ in sorted(zip(population, scores), key=lambda pair: pair[1], reverse=True)]
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


# maybe we can change it mutate by using back propagation
def mutate(population):
    for network in population:
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
