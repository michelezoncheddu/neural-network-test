from csv import reader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from network import Network


VALIDATION_MODE = True

def main():
    """The main function."""
    matplotlib.use('agg')
    features_cardinality = [3, 3, 2, 3, 4, 2]  # See data/monk.names file.
    features_values = sum(features_cardinality)
    features = len(features_cardinality)  # Number of features.
    num_epoch = 100

    # Information plot about training.
    square_errors_training = np.empty(num_epoch)
    errors_training = np.empty(num_epoch)

    # Information plot about test.
    square_errors_test = np.empty(num_epoch)
    errors_test = np.empty(num_epoch)
    epoch = np.arange(num_epoch)

    data_path = '../data/'
    training_set_path = data_path + 'training/monks-2.train'
    test_set_path = data_path + 'test/monks-2.test'

    hidden_units = 3
    output_units = 1

    nn = Network([features_values, hidden_units, output_units])

    training_set = []
    validation_set = []

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

    with open(test_set_path, 'r') as file:
        file_reader = reader(file, delimiter=' ')  # File parsing.
        for line in file_reader:
            inputs = [0] * (features_values + 1)  # + 1 for class attribute.
            inputs[0] = int(line[0])  # Saving class attribute.
            offset = 0  # Offset inside encoded array.

            # One-hot encoding.
            for j in range(features):
                inputs[int(line[j + 1]) + offset] = 1
                offset += features_cardinality[j]
            validation_set.append(inputs)

    # Setting online/minibatch/batch mode.
    nn.MINIBATCH = len(training_set)

    # Training.
    for i in range(num_epoch):
        square_error, label = nn.train(training_set)
        print(square_error / len(training_set))
        square_errors_training[i] = square_error / len(training_set)
        errors_training[i] = ((len(training_set) - label) / len(training_set)) * 100

        if VALIDATION_MODE:
            mislassifications = 0
            square_error_test = 0
            for inputs in validation_set:
                square_error, label = nn.predict(inputs)
                if round(label[0]) != int(inputs[0]):
                    mislassifications += 1
                square_error_test += square_error
            square_errors_test[i] = square_error_test / len(validation_set)
            errors_test[i] = ((len(validation_set) - mislassifications) / len(validation_set)) * 100

    # Plot learning curfve.
    fig_learn, ax_learn = plt.subplots()
    fig_acc, ax_acc = plt.subplots()

    ax_learn.plot(epoch, square_errors_training, label='training')
    ax_learn.plot(epoch, square_errors_test, label='test')

    ax_acc.plot(epoch, errors_training, label='training')
    ax_acc.plot(epoch, errors_test, label='test')
    ax_acc.set(xlabel='Epochs', ylabel='Accuracy', title='Accuracy curve')

    ax_learn.set(xlabel='Epochs', ylabel='LMS', title='Learning curve')
    ax_learn.grid()
    ax_acc.grid()

    ax_learn.legend()
    ax_acc.legend()
    fig_learn.savefig('../learning_curve.png')
    fig_acc.savefig('../accuracy_curve.png')


if __name__ == '__main__':
    main()
