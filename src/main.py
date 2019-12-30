from csv import reader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from network import Network


TESTING_MODE = True

def main():
    """The main function."""
    matplotlib.use('agg')
    features_cardinality = [3, 3, 2, 3, 4, 2]  # See data/monk.names file.
    features_values = sum(features_cardinality)
    features = len(features_cardinality)  # Number of features.
    num_epoch = 1000

    square_error = np.empty(num_epoch)
    epoch = np.arange(num_epoch)

    training_set_path = 'data/training/monks-1.train'
    test_set_path = 'data/test/monks-1.test'

    hidden_units = 3
    output_units = 1

    nn = Network([features_values, hidden_units, output_units])

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

    # Setting online/minibatch/batch mode
    nn.MINIBATCH = 1

    # Training.
    for i in range(num_epoch):
        square_error[i] = nn.train(training_set) / len(training_set)

    # Plot learning curve.
    fig, ax = plt.subplots()
    ax.plot(epoch, square_error)

    ax.set(xlabel='Epochs', ylabel='LMS', title='Learning curve')
    ax.grid()
    ax.legend('TR')
    fig.savefig('learning_curve.png')
    print(square_error[-1])

    if TESTING_MODE:
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
