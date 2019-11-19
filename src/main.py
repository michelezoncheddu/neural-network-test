from csv import reader
from network import Network


def main():
    features_cardinality = [3, 3, 2, 3, 4, 2]
    features_values = sum(features_cardinality)
    features = len(features_cardinality)  # Number of features.
    training_set_path = 'data/training/monks-1.train'
    hidden_units = 2
    output_units = 1

    nn = Network(features_values, hidden_units, output_units)
    training_set = []
        
    with open(training_set_path, 'r') as file:
        file_reader = reader(file, delimiter=' ')  # File parsing.
        for line in file_reader:
            offset = 0  # Offset inside encoded array.

            # One-hot encoding.
            inputs = [0] * features_values
            for i in range(features):
                inputs[offset + int(line[i + 1]) - 1] = 1
                offset += features_cardinality[i]

            training_set.append(inputs)
    
    nn.train(training_set)


if __name__ == '__main__':
    main()
