import numpy as np
import multiprocessing as mp
from functools import partial
from GeneticAlgorithm import GeneticAlgorithm


def check_param(param, param_key, param_dict):
    param_dict[param_key] = param
    ga = GeneticAlgorithm(param_dict)
    best_network = ga.run()
    predictions = best_network.forward(param_dict['X_train'])
    count = sum(predictions[i] == int(param_dict['y_train'][i]) for i in range(len(predictions)))
    accuracy = count / len(predictions)
    return param, accuracy

def get_data(rng):
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
        y_train.append(int(training_data[i][1].strip('\n')))

    for i in range(len(test_data)):
        test_data[i] = test_data[i].split('   ')
        string = list(training_data[i][0])
        num_string = [int(i) for i in string]

        X_test.append(np.array(num_string))
        y_test.append(test_data[i][1].strip('\n'))

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

if __name__ == "__main__":
    # for multiprocessing
    mp.freeze_support()

    rng = np.random.default_rng()

    # Load the training and test data
    X_train, y_train, X_test, y_test = get_data(rng)


    # Genetic Algorithm Parameters
    NETWORK_STRUCTURE = [16, 32, 16, 1]
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 50
    MUTATION_RATE = 0.4
    REPLICATION_RATE = 0.1
    CROSSOVER_RATE = 1 - REPLICATION_RATE
    TOURNAMENT_SIZE = 45
    LEARNING_RATE = 0.08


    # write a dictionary of the above parameters
    # write a function that takes in a dictionary of parameters and runs the genetic algorithm
    param_dict = {
                "X_train": X_train, "y_train": y_train, "rng": rng ,
                "NETWORK_STRUCTURE": NETWORK_STRUCTURE, "POPULATION_SIZE": POPULATION_SIZE,
                "MAX_GENERATIONS": MAX_GENERATIONS, "MUTATION_RATE": MUTATION_RATE,
                "REPLICATION_RATE": REPLICATION_RATE, "CROSSOVER_RATE": CROSSOVER_RATE,
                "TOURNAMENT_SIZE": TOURNAMENT_SIZE, "LEARNING_RATE": LEARNING_RATE
                }

    with open('results0.csv', 'w') as res_file:
        res_file.write("NETWORK_STRUCTURE,POPULATION_SIZE,MAX_GENERATIONS,REPLICATION_RATE,CROSSOVER_RATE,MUTATION_RATE,LEARNING_RATE,TOURNAMENT_SIZE\n")

    # set the parameter to be tested, and which values to test it over
    param_key = "NETWORK_STRUCTURE"

    # set all rags within the function except for the parameter values, for multiprocessing
    fixed_check_param = partial(check_param, param_key=param_key, param_dict=param_dict)

    params = [[16, 32, 16, 1], [16, 16, 16, 1], [16, 64, 32, 1], [16, 64, 16, 1], [16, 16, 32, 1], [16, 32, 64, 32, 1]]
    results = [fixed_check_param(params[0])]

    # run genetic algorithm to train the network over several parameters
    # with mp.Pool() as executor:
    #     results = []
    #     for param, accuracy in executor.map(fixed_check_param, params):
    #         results.append((param, accuracy))

    # write the results to a file
    with open('results0.csv', 'a') as res_file:
        for result in results:
            res_file.write(f"{result[0]},{POPULATION_SIZE},{MAX_GENERATIONS},{REPLICATION_RATE},{CROSSOVER_RATE},{MUTATION_RATE},{LEARNING_RATE},{TOURNAMENT_SIZE}\n")
            res_file.write(f"best score: {result[1]}\n")

    # Save the best network to a file
    # with open('wnet0.txt', 'w') as f:
    #     for weight in best_network.weights:
    #         for i in range(weight.shape[0]):
    #             for j in range(weight.shape[1]-1):
    #                 f.write(f"{weight[i][j]},")
    #             f.write(f"{weight[i][-1]}")
    #             f.write("\n")
    #         f.write("end of layer\n")