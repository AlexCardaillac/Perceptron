import numpy as np

def binary_step(value):
    if value < 0:
        return 0
    return 1

class Perceptron(object):

    def __init__(self, nb_inputs, activation_fnct, max_epochs=1000, learning_constant=0.01, error_threshold=0):
        self.nb_inputs = nb_inputs
        self.activation_fnct = activation_fnct
        self.max_epochs = max_epochs
        self.learning_constant = learning_constant
        self.error_threshold = error_threshold
        self.weights = np.concatenate((np.random.uniform(-1, 1, nb_inputs), [1]))

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[:-1]) + self.weights[-1]
        return self.activation_fnct(summation)

    def train(self, source_inputs, target_values, verbose=False):
        error_total = self.error_threshold + 1
        for i in (_ for _ in range(self.max_epochs) if error_total > self.error_threshold):
            error_total = 0
            for inputs, target_value in zip(source_inputs, target_values):
                output_value = self.predict(inputs)
                error_value = target_value - output_value
                self.weights[:-1] += self.learning_constant * error_value * inputs
                self.weights[-1] += self.learning_constant * error_value
                error_total += abs(error_value)
        if verbose:
            self.print_results(i, error_total)

    def print_results(self, nb_epochs, error_value):
        print('Trained in %d epochs with an error value of %f' %(nb_epochs, error_value))
        print('weights: ', end='')
        print(self.weights[:-1])
        print('bias: %f' %self.weights[-1])

        if self.nb_inputs != 2:
            print('Warning, ploting only works with 2 inputs')
        else:
            import matplotlib.pyplot as plt

            b = p1.weights[-1]
            x = -1 * b / p1.weights[0]
            y = -1 * b / p1.weights[1]
            m = (0 - y) / (x - 0)

            x_min = source_inputs.min(axis=0)[1] - 1
            x_max = source_inputs.max(axis=0)[1] + 1

            pts = [m*x_v+y for x_v in range(x_min, x_max + 1)]
            plt.plot(range(x_min, x_max + 1), pts)

            x_a = []
            y_a = []
            x_b = []
            y_b = []
            for target_value, source_input in zip(target_values, source_inputs):
                if target_value == 0:
                    y_a.append(source_input[0])
                    x_a.append(source_input[1])
                else:
                    y_b.append(source_input[0])
                    x_b.append(source_input[1])

            plt.plot(x_a, y_a, 'o', color='green')
            plt.plot(x_b, y_b, 'o', color='red')
            plt.show()
