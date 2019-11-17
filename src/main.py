from network import Network


def main():
    training_set_path = 'data/training/monks-1.train'

    # Test.
    nn = Network(6, 2, 1)

    with open(training_set_path, 'r') as file:
        nn.train(file)


if __name__ == '__main__':
    main()
