# Perceptron
Rosenblatt's Perceptron in Python

## Requirements
- python3
- numpy
- unittest

## Run tests
python3 perceptron_test.py

## How to use
You first need to prepare your data. In this example the Perceptron will learn the AND operation.
```
target_values = np.array([0, 0, 0, 1])
source_inputs = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
```
You then need to create a Perceptron with a given number of input and an activation function
```
p = Perceptron(2, binary_step)
```
_note that you can specify the maximum number of iterations, the learning constant and the error threshold._  
  
You can now train the Perceptron and make it predict an output given an input:
```
perceptron.train(source_inputs, target_values)

output = perceptron.predict([1, 1])
print(output)
```
The output should be 1.
