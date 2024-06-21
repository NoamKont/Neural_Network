import numpy as np
import pandas as pd
from Neural_Network import NeuralNetwork

if __name__ == "__main__":
    with open("..\MNIST-train.csv") as file:
        df = pd.read_csv(file)
    # Generate dataset
    y_train = df['y']
    X_train = df.drop('y', axis=1)

    with open("..\MNIST-test.csv") as file:
        df = pd.read_csv(file)
    y_test = df['y']
    X_test = df.drop('y', axis=1)

    X_train = X_train.values
    y_train = np.array(y_train)

    X_test = X_test.values
    y_test = np.array(y_test)

    # Initialize neural network
    nn = NeuralNetwork(input_size=X_train.shape[1], hidden_layer_sizes=[30, 30], output_size=10, multy_class=True)

    # set the activation function:
    nn.activation_functions = [nn.relu, nn.tanh, nn.softmax]

    # normalize the data
    X_train = nn.normalize(X_train)
    X_test = nn.normalize(X_test)

    # Train the neural network
    nn.fit(X_train, y_train, epochs=5, batch_size=1)

    # Get predictions for all samples
    print("score: ", nn.score(X_test, y_test))
    print('Predictions:', nn.predict(X_test))
    print("Actual Targets:", y_test)
