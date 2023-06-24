
if __name__ == "__main__":
    # open testnet0_actual_labels.txt and predictions0.txt, calculate accuracy
    with open("testnet1_actual_labels.txt", "r") as actual_labels_file:
        actual_labels = actual_labels_file.readlines()

    with open("predictions1.txt", "r") as predictions_file:
        predictions = predictions_file.readlines()

    correct = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == predictions[i]:
            correct += 1

    print(f"Accuracy: {correct / len(actual_labels)}")