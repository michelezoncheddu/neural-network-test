from csv import reader
from network import Network


def main():
    features_cardinality = [3, 3, 2, 3, 4, 2]  # See data/monk.names file.
    features_values = sum(features_cardinality)
    features = len(features_cardinality)  # Number of features.

    training_set_path = 'data/training/monks-2.train'
    test_set_path = 'data/test/monks-2.test'

    hidden_units = 3
    output_units = 1

    nn = Network(features_values, hidden_units, output_units)

    training_set = []
        
    with open(training_set_path, 'r') as file:
        file_reader = reader(file, delimiter=' ')  # File parsing.
        for line in file_reader:
            inputs = [0] * (features_values + 1)  # + 1 for class attribute.
            inputs[0] = int(line[0])  # Saving class attribute.
            offset = 0  # Offset inside encoded array.

            # One-hot encoding.
            for i in range(features):
                inputs[int(line[i + 1]) + offset] = 1
                offset += features_cardinality[i]

            training_set.append(inputs)

    for i in range(500):
        nn.train(training_set)
    
    errors = 0

    with open(test_set_path, 'r') as file:
        file_reader = reader(file, delimiter=' ')  # File parsing.
        for line in file_reader:
            inputs = [0] * (features_values + 1)  # + 1 for class attribute.
            inputs[0] = int(line[0])  # Saving class attribute.
            offset = 0  # Offset inside encoded array.

            # One-hot encoding.
            for i in range(features):
                inputs[int(line[i + 1]) + offset] = 1
                offset += features_cardinality[i]
            
            result = nn.predict(inputs[1:])
            if round(result[0]) != int(line[0]):
                errors += 1
        
        print(errors)


if __name__ == '__main__':
    main()
