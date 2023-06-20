import numpy as np
import pandas as pd
import multiprocessing as mp
from GeneticAlgorithm import GeneticAlgorithm
from copy import deepcopy
import time
import sys

def save_results(results):
    best_network = None
    best_accuracy = 0

    # write the results to a file
    with open('results0.csv', 'w') as res_file:
        # write header row, split for readability
        res_file.write("NETWORK_STRUCTURE,POPULATION_SIZE," +
                       "MAX_GENERATIONS,REPLICATION_RATE," +
                       "MUTATION_RATE," +
                       "LEARNING_RATE,TOURNAMENT_SIZE,BEST_SCORE\n")

        for accuracy, best_network, result_dict in results:
            arr_rep = str(result_dict['NETWORK_STRUCTURE']).split("[")[1].split("]")[0].split(",")
            res_file.write("".join(arr_rep) + "," +
                           f"{result_dict['POPULATION_SIZE']}," +
                           f"{result_dict['MAX_GENERATIONS']}," +
                           f"{result_dict['REPLICATION_RATE']}," +
                           f"{result_dict['MUTATION_RATE']}," +
                           f"{result_dict['LEARNING_RATE']}," +
                           f"{result_dict['TOURNAMENT_SIZE']}," +
                           f"{accuracy}\n")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_network = best_network

    # create dataframe from results file, get max accuracy row
    with open('results0.csv', 'r') as res_file:
        df = pd.read_csv(res_file)
        # get max BEST_SCORE index
        max_accuracy = df['BEST_SCORE'].idxmax()
        print(df.iloc[max_accuracy])


def check_param(param_dict, executor):
    ga = GeneticAlgorithm(param_dict)
    best_network = ga.run(executor)
    predictions = best_network.forward(param_dict['X_train'])
    count = sum(predictions[i] == int(param_dict['y_train'][i]) for i in range(len(predictions)))
    accuracy = count / len(predictions)
    return accuracy, best_network


def get_data(rng, train_path, test_path):
    # Load the training data
    training_data = open(train_path, "r").readlines()
    test_data = open(test_path, "r").readlines()

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

def get_validation_data(rng):
    validation_data = open("validation_set0.txt", "r").readlines()
    # and split it by "\t" the data is before the '\t' and the label is after the '\t'

    # shuffle the data
    rng.shuffle(validation_data)

    X_validation = []
    y_validation = []

    for i in range(len(validation_data)):
        validation_data[i] = validation_data[i].split('   ')
        string = list(validation_data[i][0])
        num_string = [int(i) for i in string]

        X_validation.append(np.array(num_string))
        y_validation.append(validation_data[i][1].strip('\n'))

    
    return np.array(X_validation, dtype=np.float64), np.array(y_validation)


if __name__ == "__main__":
    
    if len(sys.argv) == 3:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
    else:
        train_path = "training_set0.txt"
        test_path = "test_set0.txt"

    # for multiprocessing
    mp.freeze_support()

    rng = np.random.default_rng()

    # load the training and test data
    X_train, y_train, X_test, y_test = get_data(rng, train_path, test_path)
    X_validation, y_validation = get_validation_data(rng)

    # default Genetic Algorithm parameters
    NETWORK_STRUCTURE = [16, 32, 16, 1]
    POPULATION_SIZE = 150
    MAX_GENERATIONS = 50
    MUTATION_RATE = 0.7
    REPLICATION_RATE = 0.25
    CROSSOVER_RATE = 1 - REPLICATION_RATE
    TOURNAMENT_SIZE = 75
    LEARNING_RATE = 0.01

    # write a dictionary of the above parameters
    # write a function that takes in a dictionary of parameters and runs the genetic algorithm
    param_dict = {
        "X_train": X_train, "y_train": y_train, "rng": rng,
        "NETWORK_STRUCTURE": NETWORK_STRUCTURE, "POPULATION_SIZE": POPULATION_SIZE,
        "MAX_GENERATIONS": MAX_GENERATIONS, "MUTATION_RATE": MUTATION_RATE,
        "REPLICATION_RATE": REPLICATION_RATE, "TOURNAMENT_SIZE": TOURNAMENT_SIZE,
        "LEARNING_RATE": LEARNING_RATE
    }

    # set the parameter to be tested, and which values to test it over
    # param_key1 = "LEARNING_RATE"
    # params1 = [0.01, 0.05, 0.1, 0.15]
    # repeats = 2

    start = time.time()
    results = []

    with mp.Pool() as executor:
        
        ## run with several parameters
        # for param1 in params1:
        #     param_dict[param_key1] = param1
            # for param2 in params2:
            #     param_dict[param_key2] = param2
            
                ## run with repeats
                # avg_score = 0
                # for i in range(repeats):
                    # result = check_param(param_dict, executor=executor)
                    # avg_score += result[0]
                # avg_score /= repeats

        # run without repeats
        result = check_param(param_dict, executor=executor)
        avg_score = result[0]

        results.append([avg_score, result[1], deepcopy(param_dict)])

    end = time.time()
    print(f"time taken: {end - start}s")
    
    # save_results(results)

    # get the best network
    best_network = None
    best_accuracy = 0

    for accuracy, best_network, result_dict in results:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_network = best_network    

    # Save the best network weights to a file
    with open('wnet0.txt', 'w') as f:
        for weight in best_network.weights:
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1] - 1):
                    f.write(f"{weight[i][j]},")
                f.write(f"{weight[i][-1]}")
                f.write("\n")
            f.write("end of layer\n")
