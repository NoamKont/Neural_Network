import numpy as np
import pandas as pd
from Neural_Network import NeuralNetwork  # Assuming your NeuralNetwork class is defined in NeuralNetwork.py


if __name__ == "__main__":

    with open("..\Processed Wisconsin Diagnostic Breast Cancer.csv") as file:
        df = pd.read_csv(file)
    # # Generate dataset
    # y = df['y']
    # X = df.drop('y', axis=1)
    # y = [-1 if i == 0 else i for i in y]

    y = df['diagnosis']
    X = df.drop('diagnosis', axis=1)
    y = [-1 if i == 0 else i for i in y]

    X = X.values
    y = np.array(y)
    #Initialize neural network
    nn = NeuralNetwork(input_size=X.shape[1], hidden_layer_sizes=[3, 2], output_size=1, loss_function_name='MSE')

    X = nn.normalize(X)
    X_train, X_test, y_train, y_test = nn.split_data(X, y, )
    activation_functions = [nn.tanh, nn.sigmoid, nn.tanh, nn.relu, nn.tanh, nn.sigmoid]  # Example activation functions
    activation_derivatives = [nn.tanh_derivative, nn.sigmoid_derivative, nn.sigmoid_derivative, nn.sigmoid_derivative, nn.tanh_derivative, nn.sigmoid_derivative]  # Example derivatives

    nn.activation_functions = activation_functions
    nn.activation_derivatives = activation_derivatives

    # Train the neural network
    nn.fit(X_train, y_train, epochs=10, batch_size=1)

    # Get predictions for all samples
    print("score: ", nn.score(X_test, y_test))
    print('Predictions:', nn.predict(X_test))
    print("Actual Targets:", y_test)


