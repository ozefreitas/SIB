import numpy as np
from src.si.util.activation import *
from src.si.data import Dataset
from src.si.supervised.Modelo import Model


def backward(self, output_error, learning_rate):
    # dE/dW ? X.T * dE/dY
    weights_error = np.dot(self.input.T, output_error)
    # dE/dB = dE/dY
    bias_error = np.sum(output_error, axis=0)
    # dE/dX
    input_error = np.dot(output_error, self.weights.T)
    self.weights -= learning_rate * weights_error
    self.bias -= learning_rate * bias_error
    return input_error


class Layer:
    def setWeights(self, weights, bias):
        self.weights = weights
        self.bias = bias


class Activation(Layer):
    pass


class NN:
    def __init__(self):
        pass


    def fit(self, dataset):
        X, Y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            #forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(Y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            err = self.loss(Y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f"epoch {epoch+1}/{self.epochs} error={err}")

        if not self.verbose:
            print(f"error={err}")
        self.is_fited = True

    def backward(self, output_error, learning_rate):
        return np.multiply(self.activation.prime(self.input), output_error)

