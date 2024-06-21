import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size=3, hidden_layer_sizes=[5], output_size=1, loss_function_name = "MSE-G", multy_class = False):
        self.inputLayerSize = input_size
        self.outputLayerSize = output_size
        self.hiddenLayerSize = hidden_layer_sizes
        self.weights = []
        self.activations = []
        self.S = []
        self.activation_functions = []
        self.activation_derivatives = []
        self.learning_rate = 0.25
        self.thresh = 0.000001
        self.initialize_weights()
        self.loss_functions = None
        self.loss_derivative_function = None
        self.choose_loss_function(loss_function_name)
        self.batch = 25
        self.multy_class = multy_class


    #initialization the weights (random initialization)
    def initialize_weights(self):
        for i in range(len(self.hiddenLayerSize) + 1):
            if i == 0:
                self.weights.append(np.random.randn(self.inputLayerSize+1, self.hiddenLayerSize[i]-1))
            elif i == len(self.hiddenLayerSize):
                self.weights.append(np.random.randn(self.hiddenLayerSize[-1], self.outputLayerSize))
            else:
                self.weights.append(np.random.randn(self.hiddenLayerSize[i - 1], self.hiddenLayerSize[i]-1))
            self.weights[i] /= np.linalg.norm(self.weights[i]) #normlize the weights

    #initialize empty vector to calculate the norma of the gradient
    def initialize_zero(self):
        matrix_array = []
        for i in range(len(self.hiddenLayerSize) + 1):
            if i == 0:
                matrix_array.append(np.zeros((self.inputLayerSize + 1, self.hiddenLayerSize[i]-1)))
            elif i == len(self.hiddenLayerSize):
                matrix_array.append(np.zeros((self.hiddenLayerSize[-1], self.outputLayerSize)))
            else:
                matrix_array.append(np.zeros((self.hiddenLayerSize[i - 1], self.hiddenLayerSize[i]-1)))
        return np.array(matrix_array, dtype=object)

    #activisions functions and the derivative
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = NeuralNetwork.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=0)

    @staticmethod
    def softmax_derivative(x):
        return NeuralNetwork.softmax(x) - NeuralNetwork.softmax(x)**2

    def MSE_derivative(self, y, h):
        return 2*(h - y)

    def MSE_Gradient_derivative(self, y, h):
        return (h - y)/len(y)

    def MSE(self, y, h):
        return np.sum((y - h) ** 2)

    def update_activation_functions_derivative(self):
        for function in self.activation_functions:
            if function == self.sigmoid:
                self.activation_derivatives.append(self.sigmoid_derivative)
            elif function == self.relu:
                self.activation_derivatives.append(self.relu_derivative)
            elif function == self.tanh:
                self.activation_derivatives.append(self.tanh_derivative)
            elif function == self.softmax:
                self.activation_derivatives.append(self.softmax_derivative)

    def choose_loss_function(self, function_name):
        if function_name == 'MSE':
            self.loss_functions = self.MSE
            self.loss_derivative_function = self.MSE_derivative
        elif function_name == 'MSE-G':
            self.loss_derivative_function = self.MSE_Gradient_derivative

    def forward_propagation(self, x):
        x = np.append(x, [[1]])  # Add bias term
        activations = [x]  # Store the input data as the first element
        self.S = []
        for i in range(len(self.weights)):
            p = np.dot(activations[-1], self.weights[i])
            x_i = self.activation_functions[i](p)
            if i < len(self.weights) - 1:
                x_i = np.append(x_i, [[1]])
            self.S.append(p)
            activations.append(x_i)  # Store activation of the current layer
        self.activations = activations
        return activations[-1]

    def backward_propagation(self, x, y, multy_class = True):
        self.forward_propagation(x)
        deltas = []

        # Compute delta for the output layer
        if self.multy_class:
            y = self.one_hot_encode(y, self.outputLayerSize)
        error = self.loss_derivative_function(y, self.activations[-1])
        delta = error * self.activation_derivatives[-1](self.S[-1])
        deltas.append(delta)

        # Compute deltas for hidden layers (backwards)

        for i in range(len(self.weights) - 2, -1, -1):
            weights_len = self.weights[i + 1].shape[0]
            delta = np.dot(deltas[-1], (self.weights[i + 1].T)[:, :weights_len - 1]) * self.activation_derivatives[i](self.S[i])
            deltas.append(delta)

        # Reverse the deltas to match the layer order
        deltas.reverse()
        return deltas

    def fit(self, X, y, epochs=1000, batch_size=1):
        min_norma = np.inf
        norma_smaller_than_thresh = False
        best_weight = [] #paramter for pocket algorithm
        self.update_activation_functions_derivative()
        for epoch in range(epochs):
            # Shuffle the data for SGD
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            sum_of_batch_gradient = self.initialize_zero()
            # Iterate over mini-batches
            for i in range(0, len(X_shuffled), batch_size):
                # Backward pass
                deltas = self.backward_propagation(X_shuffled[i], y_shuffled[i])

                # Update weights and biases using SGD
                for j in range(len(self.weights)):
                    gradient = np.outer(self.activations[j], deltas[j])
                    sum_of_batch_gradient[j] += gradient
                    self.weights[j] -= self.learning_rate * gradient

                if (i % self.batch == 0 and i != 0):
                    norm = 0
                    for matrix in sum_of_batch_gradient:
                        norm += np.linalg.norm(matrix/self.batch)**2
                    if np.sqrt(norm) < min_norma:
                        min_norma = np.sqrt(norm)
                        best_weight = self.weights

                    if np.sqrt(norm) < self.thresh:
                        norma_smaller_than_thresh = True
                        break
                    sum_of_batch_gradient = self.initialize_zero()
            if norma_smaller_than_thresh:
                break
        self.weights = best_weight

    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            p = self.forward_propagation(X[i])  # p stand for predicted
            if self.multy_class:
                pred.append(np.argmax(p))
            else:
                pred.append(np.sign(p[0]))
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        return np.sum(pred == y)/len(y)

    def one_hot_encode(self, value, size=10):
        vector = np.zeros(size)
        vector[value] = 1
        return vector


    @staticmethod
    def split_data(X, y, training_size=0.8):
        length = len(X)
        num_train = int(length * training_size)

        X_train = X[:num_train]
        X_test = X[num_train:]

        y_train = y[:num_train]
        y_test = y[num_train:]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def normalize(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        # Avoid division by zero
        std[std == 0] = 1

        X_normalized = (X - mean) / std
        return X_normalized


