import numpy as np
from NeuralNetwork import NeuralNetwork
class GeneticAlgorithm:
    def __init__(self, X_train, y_train, rng, sizes, population_size, max_generations,
                  replication_rate, crossover_rate, mutation_rate,
                    learning_rate, tournament_size):
        self.X_train = X_train
        self.y_train = y_train
        self.sizes = sizes
        self.rng = rng
        self.population_size = population_size
        self.max_generations = max_generations
        self.replication_rate = replication_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.tournament_size = tournament_size

    
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
            population = np.concatenate((elite_population, crossover_population))
            self.mutate(population)

        best_network = population[0]['net']
        return best_network


    def evaluate_and_sort(self, population):
        for i in range(self.population_size):
            predictions = population[i]['net'].forward(self.X_train)
            count = sum(predictions[i] == int(self.y_train[i]) for i in range(len(predictions)))
            accuracy = count / len(predictions)
            population[i]['score'] = accuracy

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
                if random_assignments[j] == 0:
                    child.weights[j] = np.copy(parent1['net'].weights[j])
                else:
                    child.weights[j] = np.copy(parent2['net'].weights[j])
            
            offsprings[i]['score'] = 0
            offsprings[i]['net'] = child

        return offsprings

    def mutate(self, networks):
        for net_index in range(1, self.population_size):
            for i in range(len(networks[net_index]['net'].weights)):
                    if self.rng.random() < self.mutation_rate:
                        networks[net_index]['net'].weights[i] += (self.rng.standard_normal(size=networks[net_index]['net'].weights[i].shape) *
                            self.learning_rate)
            