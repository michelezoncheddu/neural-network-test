from csv import reader
from network import Network


# TODO: replace ALL lists in the project with arrays


def main():
    features_cardinality = [2, 3, 3, 2, 3, 4, 2]  # See data/monk.names file.
    features_values = sum(features_cardinality)
    features = len(features_cardinality)  # Number of features.

    training_set_path = 'data/training/monks-2.train'

    hidden_units = 2
    output_units = 2

    nn = Network(features_values - features_cardinality[0], hidden_units, output_units)

    training_set = []
    
    with open(training_set_path, 'r') as file:
        file_reader = reader(file, delimiter=' ')  # File parsing.
        for line in file_reader:
            inputs = [0] * features_values

            inputs[int(line[0])] = 1  # Saving classification attribute.

            offset = features_cardinality[0]  # Offset inside encoded array.

            # One-hot encoding.
            for i in range(1, features):
                inputs[int(line[i]) + offset - 1] = 1
                offset += features_cardinality[i]

            training_set.append(inputs)

    for i in range(10):
        nn.train(training_set)


if __name__ == '__main__':
    main()
