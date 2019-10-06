import numpy as np

def binary_step(value):
    if value < 0:
        return 0
    return 1

class Perceptron(object):

    def __init__(self, nb_inputs, activation_fnct, max_epochs=1000, learning_constant=0.01, error_threshold=0.1):
        self.nb_inputs = nb_inputs
        self.activation_fnct = activation_fnct
        self.max_epochs = max_epochs
        self.learning_constant = learning_constant
        self.error_threshold = error_threshold
        self.weights = np.concatenate((np.random.uniform(-1, 1, nb_inputs), [1]))

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[:-1]) + self.weights[-1]
        return self.activation_fnct(summation)

    def train(self, source_inputs, target_values):
        error_total = self.error_threshold + 1
        for _ in (_ for _ in range(self.max_epochs) if error_total > self.error_threshold):
            error_total = 0
            for inputs, target_value in zip(source_inputs, target_values):
                output_value = self.predict(inputs)
                error_value = target_value - output_value
                self.weights[:-1] += self.learning_constant * error_value * inputs
                self.weights[-1] += self.learning_constant * error_value
                error_total += abs(error_value)
