import numpy as np
import numba as nb
from NeuralNetwork import NeuralNetwork
import time

@nb.jit(nopython=True, cache=True)
def jit_forward(X_train, y_train, weights):
    """forward propogate the input through the networks

    Args:
        inputs (nd.array): input data
        networks (nd.array): array of network scores and classes

    Returns:
        nd.array: predictions
    """
    x = X_train.copy()
    for weight in weights:
        hidden = np.dot(x, weight)
        x = 1 / (1 + np.exp(-hidden))
    
    answer = x.flatten()
    predictions = np.round(answer)
    accuracy = sum([predictions[i] == int(y_train[i]) for i in range(len(y_train))]) / len(y_train)

    return accuracy

class GeneticAlgorithm:
    def __init__(self, param_dict):
        self.X_train = param_dict['X_train']
        self.y_train = param_dict['y_train']
        self.rng = param_dict['rng']
        self.sizes = param_dict['NETWORK_STRUCTURE']
        self.population_size = param_dict['POPULATION_SIZE']
        self.max_generations = param_dict['MAX_GENERATIONS']
        self.replication_rate = param_dict['REPLICATION_RATE']
        self.crossover_rate = param_dict['CROSSOVER_RATE']
        self.mutation_rate = param_dict['MUTATION_RATE']
        self.learning_rate = param_dict['LEARNING_RATE']
        self.tournament_size = param_dict['TOURNAMENT_SIZE']
    
        
    def single_forward(self, X_train, y_train, network):
        """forward propogate the input through the networks

        Args:
            inputs (nd.array): input data
            networks (nd.array): array of network scores and classes

        Returns:
            nd.array: predictions
        """
        # x = np.copy(X_train)
        # for weight in network.weights:
        #     hidden = np.dot(x, weight)
        #     x = network.sigmoid(hidden)
        
        # answer = x.flatten()
        # predictions = np.where(answer > 0.5, 1, 0)
        # predictions -= y_train
        # accuracy = np.count_nonzero(predictions) / len(predictions)
        weights = nb.typed.List(network.weights)
        accuracy = jit_forward(X_train, y_train, weights)

        return accuracy
    

    def run(self):
        population = np.array([(0, NeuralNetwork(self.rng, self.sizes)) for i in range(self.population_size)],
                          dtype=[('score', float), ('net', NeuralNetwork)])

        for generation in range(self.max_generations):
            # print(f"Generation: {generation + 1}")
            sorted_population_scores = self.evaluate_and_sort(population)

            # print the best network in the generation
            # print(f"Best network: {sorted_population_scores[0]['net']}")
            # print the best score in the generation
            # print(f"Best score: {sorted_population_scores[0]['score']}")

            elite_population = self.select(sorted_population_scores)
            crossover_population = self.crossover(sorted_population_scores)
            # crossover pop is mutated in the crossover function for efficiency
            self.mutate(elite_population)
            population = np.concatenate((elite_population, crossover_population))
            

        best_network = population[0]['net']
        return best_network


    def evaluate_and_sort(self, population):
        # vectorize the forward function to increase code efficiency
        vec_forward = np.vectorize(self.single_forward, excluded=['X_train', 'y_train'])
        population['score'] = vec_forward(X_train=self.X_train, y_train=self.y_train, network=population['net'])

        population = np.sort(population, order='score')
        # print("before:")
        # print(population['score'])
        population = population[::-1]
        # print("after:")
        # print(population['score'])
        return population


    def select(self, sorted_population):
        elite_count = int(self.replication_rate * self.population_size)
        return sorted_population[:elite_count]


    def crossover(self, sorted_population):
        cross_size = int(self.population_size * self.crossover_rate)
        offsprings = np.array([(0, None) for i in range(cross_size)], dtype=[('score', float), ('net', NeuralNetwork)])
        tournament_winners = np.array([(0, None) for i in range(cross_size)],
                                    dtype=[('score', float), ('net', NeuralNetwork)])

        total_score = sorted_population['score'].sum()
        sorted_population['score'] = sorted_population['score'] / total_score
        for i in range(cross_size):
            tournament_scores = self.rng.choice(sorted_population, p=sorted_population['score'], size=self.tournament_size)
            tournament_scores = np.sort(tournament_scores, order='score')
            tournament_winners[i] = tournament_scores[-1]

        for i in range(cross_size):
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
                        networks[net_index]['net'].weights[i] += (self.rng.standard_normal(size=networks[net_index]['net'].weights[i].shape) *
                            self.learning_rate)
            