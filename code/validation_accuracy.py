import sys

if __name__ == "__main__":
    
    file_num = 0
    # get arguments
    if len(sys.argv) > 1:
        file_num = sys.argv[1]

    # open testnet0_actual_labels.txt and predictions0.txt, calculate accuracy
    with open(f"testnet{file_num}_actual_labels.txt", "r") as actual_labels_file:
        actual_labels = actual_labels_file.readlines()

    with open(f"predictions{file_num}.txt", "r") as predictions_file:
        predictions = predictions_file.readlines()

    correct = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == predictions[i]:
            correct += 1

    print(f"Accuracy: {correct / len(actual_labels)}")