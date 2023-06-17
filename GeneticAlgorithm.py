import numpy as np
from NeuralNetwork import NeuralNetwork
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

    def run(self):
        population = np.array([(0, NeuralNetwork(self.rng, self.sizes)) for i in range(self.population_size)],
                          dtype=[('score', float), ('net', NeuralNetwork)])

        for generation in range(self.max_generations):
            print(f"Generation: {generation + 1}")
            sorted_population_scores = self.evaluate_and_sort(population)
            # print the best network in the generation
            print(f"Best network: {sorted_population_scores[0]['net']}")
            # print the best score in the generation
            print(f"Best score: {sorted_population_scores[0]['score']}")

            elite_population = self.select(sorted_population_scores)
            crossover_population = self.crossover(sorted_population_scores)
            self.mutate(elite_population)
            population = np.concatenate((elite_population, crossover_population))
            

        best_network = population[0]['net']
        return best_network


    def evaluate_and_sort(self, population):
        for i in range(self.population_size):
            predictions = population[i]['net'].forward(self.X_train)
            count = sum(predictions[i] == int(self.y_train[i]) for i in range(len(predictions)))
            accuracy = count / len(predictions)
            population[i]['score'] = accuracy

        # predictions = population['net'].forward(self.X_train)
        # count = np.sum(predictions == self.y_train.reshape(-1, 1), axis=1)
        # accuracy = count / len(predictions)
        # population['score'] = accuracy

        population = np.sort(population, order='score')
        print("before:")
        print(population['score'])
        population = population[::-1]
        print("after:")
        print(population['score'])
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
            