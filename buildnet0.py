import numpy as np
import multiprocessing as mp
from GeneticAlgorithm import GeneticAlgorithm
from copy import deepcopy
import time

def check_param(param_dict, executor):
    ga = GeneticAlgorithm(param_dict)
    best_network = ga.run(executor)
    predictions = best_network.forward(param_dict['X_train'])
    count = sum(predictions[i] == int(param_dict['y_train'][i]) for i in range(len(predictions)))
    accuracy = count / len(predictions)
    return accuracy, best_network

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

    return np.array(X_train, dtype=np.float64), np.array(y_train), np.array(X_test, dtype=np.float64), np.array(y_test)

# def timed_run(param):
#     # start = time.time()
#     results = [fixed_check_param(param)]
#     # end = time.time()
#     # print(f"time taken: {end - start}s")
#     return results

if __name__ == "__main__":
    # for multiprocessing
    mp.freeze_support()

    rng = np.random.default_rng()

    # load the training and test data
    X_train, y_train, X_test, y_test = get_data(rng)


    # default Genetic Algorithm parameters
    NETWORK_STRUCTURE = [16, 32, 16, 1]
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 50
    MUTATION_RATE = 0.4
    REPLICATION_RATE = 0.1
    CROSSOVER_RATE = 1 - REPLICATION_RATE
    TOURNAMENT_SIZE = 60
    LEARNING_RATE = 0.08


    # write a dictionary of the above parameters
    # write a function that takes in a dictionary of parameters and runs the genetic algorithm
    param_dict = {
                "X_train": X_train, "y_train": y_train, "rng": rng ,
                "NETWORK_STRUCTURE": NETWORK_STRUCTURE, "POPULATION_SIZE": POPULATION_SIZE,
                "MAX_GENERATIONS": MAX_GENERATIONS, "MUTATION_RATE": MUTATION_RATE,
                "REPLICATION_RATE": REPLICATION_RATE,"TOURNAMENT_SIZE": TOURNAMENT_SIZE,
                 "LEARNING_RATE": LEARNING_RATE
                }

    # set the parameter to be tested, and which values to test it over
    param_key1 = "NETWORK_STRUCTURE"
    param_key2 = "MUTATION_RATE"
    param_key3 = "REPLICATION_RATE"
    params1 = [[16, 32, 16, 1], [16, 16, 16, 1],
                [16, 64, 32, 1], [16, 64, 16, 1],
                  [16, 16, 32, 1], [16, 32, 64, 32, 1]]
    params2 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    params3 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


    start = time.time()

    results = []
    with mp.Pool() as executor:
        for param1 in params1:
            param_dict[param_key1] = param1
            for param2 in params2:
                param_dict[param_key2] = param2
                for param3 in params3:
                    param_dict[param_key3] = param3
                    # set all rags within the function except for the parameter values, for multiprocessing
                    result = check_param(param_dict, executor=executor)
                    results.append([result[0], result[1], deepcopy(param_dict)])

    end = time.time()
    print(f"time taken: {end - start}s")
    best_network = None
    best_accuracy = 0
    # write the results to a file
    with open('results0.csv', 'w') as res_file:
        # write header row, split for readability
        res_file.write("NETWORK_STRUCTURE,POPULATION_SIZE," +
                       "MAX_GENERATIONS,REPLICATION_RATE," +
                       "MUTATION_RATE," +
                       "LEARNING_RATE,TOURNAMENT_SIZE\n")
        
        for accuracy, best_network, result_dict in results:
            res_file.write(f"{result_dict['NETWORK_STRUCTURE']}," +
                           f"{result_dict['POPULATION_SIZE']}," +
                           f"{result_dict['MAX_GENERATIONS']}," +
                           f"{result_dict['REPLICATION_RATE']}," +
                           f"{result_dict['MUTATION_RATE']}," +
                           f"{result_dict['LEARNING_RATE']}," +
                           f"{result_dict['TOURNAMENT_SIZE']}\n")
            
            res_file.write(f"best score: {accuracy}\n")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_network = best_network

    # Save the best network weights to a file
    with open('wnet0.txt', 'w') as f:
        for weight in best_network.weights:
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]-1):
                    f.write(f"{weight[i][j]},")
                f.write(f"{weight[i][-1]}")
                f.write("\n")
            f.write("end of layer\n")