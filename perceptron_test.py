import unittest
import numpy as np
from perceptron import Perceptron, binary_step

class PerceptronTest(unittest.TestCase):

    def test_logical_and(self):
        target_values = np.array([0, 0, 0, 1])
        source_inputs = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))

        perceptron = Perceptron(2, binary_step)
        perceptron.train(source_inputs, target_values)

        output = perceptron.predict([1, 1])
        self.assertEqual(output, 1 & 1)
        output = perceptron.predict([0, 1])
        self.assertEqual(output, 0 & 1)
        output = perceptron.predict([1, 0])
        self.assertEqual(output, 1 & 0)
        output = perceptron.predict([0, 0])
        self.assertEqual(output, 0 & 0)

    def test_logical_or(self):
        target_values = np.array([0, 1, 1, 1])
        source_inputs = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))

        perceptron = Perceptron(2, binary_step)
        perceptron.train(source_inputs, target_values)

        output = perceptron.predict([1, 1])
        self.assertEqual(output, 1 | 1)
        output = perceptron.predict([0, 1])
        self.assertEqual(output, 0 | 1)
        output = perceptron.predict([1, 0])
        self.assertEqual(output, 1 | 0)
        output = perceptron.predict([0, 0])
        self.assertEqual(output, 0 | 0)

class ActivationFunctionTest(unittest.TestCase):

    def test_logical_or(self):
        output = binary_step(1)
        self.assertEqual(output, 1)
        output = binary_step(0)
        self.assertEqual(output, 1)
        output = binary_step(-1)
        self.assertEqual(output, 0)

if __name__ == '__main__':
        unittest.main()
