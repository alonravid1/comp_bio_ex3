# Load the training data
with open("nn0.txt", "r") as data_file:
    data = data_file.readlines()

data_size = 18000
validation_size = 2000
training_size = int(data_size * 0.8)
test_size = int(data_size * 0.2)

with open("training_set.txt", "w") as training_file:
    for line in data[:training_size]:
        training_file.write(line)

with open("test_set.txt", "w") as test_file:
    for line in data[training_size:data_size]:
        test_file.write(line)

with open("validation_set.txt", "w") as validation_file:
    for line in data[data_size:]:
        validation_file.write(line)