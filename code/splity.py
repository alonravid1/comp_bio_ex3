import sys

if __name__ == "__main__":

    file_num = 0
    # get arguments
    if len(sys.argv) > 1:
        file_num = sys.argv[1]

    # Load the training data
    with open(f"nn{file_num}.txt", "r") as data_file:
        data = data_file.readlines()

    data_size = 18000
    validation_size = 2000
    training_size = int(data_size * 0.8)
    test_size = int(data_size * 0.2)

    with open(f"training_set{file_num}.txt", "w") as training_file:
        for line in data[:training_size]:
            training_file.write(line)

    with open(f"test_set{file_num}.txt", "w") as test_file:
        for line in data[training_size:data_size]:
            test_file.write(line)

    with open(f"testnet{file_num}.txt", "w") as validation_file:
        for line in data[data_size:]:
            # get line until first tab
            line = line.split("   ")[0]
            validation_file.write(line + "\n")

    with open(f"testnet{file_num}_actual_labels.txt", "w") as validation_file:
        for line in data[data_size:]:
            # get line until first tab
            line = line.split("   ")[1]
            validation_file.write(line)