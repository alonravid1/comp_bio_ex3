import numpy as np
import numba as nb
from NeuralNetwork import NeuralNetwork
import multiprocessing as mp
from functools import partial
import time

def plot_scores(scores):
    import matplotlib.pyplot as plt
    plt.plot(scores)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Improvement Over Generations, NN1")
    plt.legend(["Best Network", "Average Network"])
    plt.show()

@nb.jit(nopython=True, cache=True)
def jit_forward(X_train, y_train, weights):
    """forward propogate the input through the networks,
    jit copmiles the function to machine code for faster execution

    Args:
        inputs (nd.array): input data
        networks (nd.array): array of network scores and classes

    Returns:
        nd.array: predictions
    """
    x = X_train.copy()
    for weight in weights:
        hidden = np.dot(x, weight)
        # sigmoid activation function, numba doesnt work with class functions
        x = 1 / (1 + np.exp(-hidden))

    answer = x.flatten()
    predictions = np.round(answer)
    accuracy = sum([predictions[i] == int(y_train[i]) for i in range(len(y_train))]) / len(y_train)

    return accuracy

def single_forward(network, i, X_train, y_train):
    """forward propogate the input through the networks,
    the function takes the NueralNetwork object and the input data
    and sends it as arrays so numba can compile it without
    issues for much faster execution

    Args:
        network (_type_): _description_
        i (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_

    Returns:
        _type_: _description_
    """
    weights = nb.typed.List(network['net'].weights)
    accuracy = jit_forward(X_train, y_train, weights)
    return accuracy, i


class GeneticAlgorithm:
    def __init__(self, param_dict):
        self.X_train = param_dict['X_train']
        self.y_train = param_dict['y_train']
        self.rng = param_dict['rng']
        self.sizes = param_dict['NETWORK_STRUCTURE']
        self.population_size = int(param_dict['POPULATION_SIZE'])
        self.max_generations = param_dict['MAX_GENERATIONS']
        self.replication_rate = float(param_dict['REPLICATION_RATE'])
        self.crossover_rate = 1 - self.replication_rate
        self.mutation_rate = param_dict['MUTATION_RATE']
        self.learning_rate = param_dict['LEARNING_RATE']
        self.tournament_size = param_dict['TOURNAMENT_SIZE']

        # this makes sure the replication and crossover rates are even numbers,
        # preventing population size from changing
        self.replication_size = (int(self.population_size * self.replication_rate) +
                                int(self.population_size * self.replication_rate) % 2)
        self.cross_size = self.population_size - self.replication_size

    def run(self, executor):
        """run the genetic algorithm

        Args:
            executor (multiprocessing.Pool): a pool of processes to run the algorithm in parallel

        Returns:
            _type_: _description_
        """
        population = np.array([(0, NeuralNetwork(self.rng, self.sizes)) for i in range(self.population_size)],
                          dtype=[('score', float), ('net', NeuralNetwork)])

        self. executor = executor
        scores = []
        for generation in range(self.max_generations):
            # print(f"Generation: {generation + 1}")

            # evaluate the population and sort it by score, efficiency
            # changes were made mostly here
            sorted_population_scores = self.evaluate_and_sort(population)

            # print the best network in the generation
            # print(f"Best network: {sorted_population_scores[0]['net']}")
            # print the best score in the generation
            avg_score = sorted_population_scores['score'].sum() / self.population_size
            scores.append((sorted_population_scores[0]['score'], avg_score))

            elite_population = self.select(sorted_population_scores)
            crossover_population = self.crossover(sorted_population_scores)

            # crossover pop is mutated in the crossover function for efficiency
            self.mutate(elite_population)

            population = np.concatenate((elite_population, crossover_population))


        best_network = population[0]['net']
        # plot_scores(scores)
        return best_network


    def evaluate_and_sort(self, population):
        """
        evaluate the networks in the population and sort them by score
        
        Args:
            population (nd.array): array of network scores and classes
        Returns:
            _type_: _description_
        """

        # start = time.time() # run time testing

        # prepare the arguments for the pool
        iter_args = [(population[i], i) for i in range(self.population_size)]

        # fix the arguments for the pool, so X_train and y_train dont have to be copied 100 times
        fixed_single_forward = partial(single_forward, X_train=self.X_train, y_train=self.y_train)

        # parallel execution of the forward propogation for the generation
        for score, i in self.executor.starmap(fixed_single_forward, iter_args):
            population[i]['score'] = score

        # end = time.time() # run time testing
        # print(f"time taken: {end - start}s") # run time testing

        sorted_population_scores = np.sort(population, order='score')[::-1]

        # print("before:")
        # print(population['score'])
        # population = population[::-1]
        # print("after:")
        # print(population['score'])

        return sorted_population_scores


    def select(self, sorted_population):
        return sorted_population[:self.replication_size]


    def crossover(self, sorted_population):
        offsprings = np.array([(0, None) for i in range(self.cross_size)], dtype=[('score', float), ('net', NeuralNetwork)])
        tournament_winners = np.array([(0, None) for i in range(self.cross_size)],
                                    dtype=[('score', float), ('net', NeuralNetwork)])

        total_score = sorted_population['score'].sum()
        sorted_population['score'] = sorted_population['score'] / total_score

        for i in range(self.cross_size):
            tournament_scores = self.rng.choice(sorted_population, p=sorted_population['score'], size=self.tournament_size)
            tournament_scores = np.sort(tournament_scores, order='score')
            tournament_winners[i] = tournament_scores[-1]

        for i in range(self.cross_size):
            parent1 = self.rng.choice(tournament_winners)  # Extract the network object from the tuple
            parent2 = self.rng.choice(tournament_winners)  # Extract the network object from the tuple
            child = NeuralNetwork(self.rng, self.sizes)
            random_assignments = self.rng.integers(0, 2, size=(len(self.sizes)-1))

            for j in range(len(random_assignments)):
                # add mutation to crossover product children here to increase code efficiency
                if self.rng.random() < self.mutation_rate:
                    if random_assignments[j] == 0:
                        child.weights[j] = (np.copy(parent1['net'].weights[j]) +
                        (self.rng.standard_normal(size=parent1['net'].weights[j].shape) * self.learning_rate))
                    else:
                        child.weights[j] = (np.copy(parent2['net'].weights[j]) +
                        (self.rng.standard_normal(size=parent2['net'].weights[j].shape) * self.learning_rate))
                else:
                    if random_assignments[j] == 0:
                        child.weights[j] = np.copy(parent1['net'].weights[j])
                    else:
                        child.weights[j] = np.copy(parent2['net'].weights[j])

            offsprings[i]['score'] = 0
            offsprings[i]['net'] = child

        return offsprings

    def mutate(self, networks):
        """mutate the weights of the elite networks

        Args:
            networks (nd.array): array of network scores and classes
        """
        for net_index in range(1, networks.shape[0]):
            for i in range(len(networks[net_index]['net'].weights)):
                    if self.rng.random() < self.mutation_rate:
                        networks[net_index]['net'].weights[i] += (self.rng.normal(size=networks[net_index]['net'].weights[i].shape) *
                            self.learning_rate)
