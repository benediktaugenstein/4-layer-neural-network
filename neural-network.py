import numpy as np
import pandas as pd

# Makes sure 'random' numbers are the same each time (optional)
# If unset -> Numbers are completely random
#np.random.seed(0)

# Choose dataset from Excel
file = pd.ExcelFile('data_python.xlsx')
dataset = file.parse('data')
array1 = dataset.values
array1 = np.array(array1)

# Set input
input_data = array1.astype(float)

# Print input (optional)
print("DATASET INPUT:")
print(input_data)

# Set targets (0 or 1 in this case; 0 -> no default, 1 -> default)
dataset = file.parse('default')
target = dataset.values
target = np.array(target)
target = target.astype(float)


# Print targets (optional)
#print("TARGET:")
#print(target)

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:

    def __init__(self, n_inputs, n_neurons):
        # random weights created, biases = 0
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs_new):
        # multiplies input matrix with weights and adds biases
        self.output = sigmoid(np.dot(inputs_new, self.weights) + self.biases)

    def backpropagation(self):
        # algorithm to change weights and biases in a specific way to achieve predictive accuracy (learning process)
        d_weights_l4 = ((self.output - target) * sigmoid_derivative(self.output))
        d_biases_l4 = (-1) * np.sum(d_weights_l4, axis=0)
        self.weights += 0.1 * (-1) * np.dot(layer3.output.T, d_weights_l4)
        self.biases += 0.1 * d_biases_l4

        d_weights_l3 = np.dot(d_weights_l4, layer4.weights.T) * sigmoid_derivative(layer3.output)
        d_weights_l2 = np.dot(d_weights_l3, layer3.weights.T) * sigmoid_derivative(layer2.output)
        d_weights_l1 = np.dot(d_weights_l2, layer2.weights.T) * sigmoid_derivative(layer1.output)

        layer3.weights += 0.1 * (-1) * np.dot(layer2.output.T, d_weights_l3)
        layer2.weights += 0.1 * (-1) * np.dot(layer1.output.T, d_weights_l2)
        layer1.weights += 0.1 * (-1) * np.dot(input_data.T, d_weights_l1)

        d_biases_l3 = (-1) * np.sum(d_weights_l3, axis=0)
        d_biases_l2 = (-1) * np.sum(d_weights_l2, axis=0)
        d_biases_l1 = (-1) * np.sum(d_weights_l1, axis=0)

        layer3.biases = 0.1 * layer3.biases + d_biases_l3
        layer2.biases = 0.1 * layer2.biases + d_biases_l2
        layer1.biases = 0.1 * layer1.biases + d_biases_l1


# NeuralNetwork(number of inputs, number of neurons):
# Generation of weights and biases
# number of inputs for layer 1 has to be the same as the number of columns in the data table.
# number of neurons is going to be the number of outputs for each layer.
# The number of inputs for each layer has to be the same as the number of neurons (outputs) from the previous layer.
layer1 = NeuralNetwork(3, 4)
layer2 = NeuralNetwork(4, 6)
layer3 = NeuralNetwork(6, 3)
layer4 = NeuralNetwork(3, 1)

# calls forward function; specific input has to be defined in ().
layer1.forward(input_data)
layer2.forward(layer1.output)
layer3.forward(layer2.output)
layer4.forward(layer3.output)

# Learning Process:
# repeat the learning process for x learning cycles
learning_cycles = 1000
for i in range(learning_cycles):
    layer4.backpropagation()
    layer1.forward(input_data)
    layer2.forward(layer1.output)
    layer3.forward(layer2.output)
    layer4.forward(layer3.output)

# Set input to make predictions for (Test data from Excel)
new_input_data = np.array([[2.5, 3.5, 3.5],
                           [5.5, 2.5, 5.5],
                           [-3.5, -1, 2.5],
                           [5.5, -2.5, 5.5],
                           [-3.5, 3.5, 5.5],
                           [-5.5, 2.5, 3.5],
                           [-1, 3.5, 2.5],
                           [1, 1, 1],
                           [5.5, -2.5, 3.5],
                           [-1, 1, -3.5],
                           [-2, -2.5, 2.5],
                           [-3.5, 1, 1],
                           [1, -1, -5.5],
                           [1, -3.5, -1],
                           [-3.5, 2.5, 5.5],
                           [5.5, -1, -5.5],
                           [1, 2.5, -2.5],
                           [-5.5, -2.5, 1],
                           [5.5, -2.5, -5.5],
                           [3.5, -2.5, -5.5],
                           [-1, 1, 3.5],
                           [5.5, -1, -5.5],
                           [3.5, -3.5, -5.5]])

# forward input (output is going to be calculated with updated "learned" weights and biases)
layer1.forward(new_input_data)
layer2.forward(layer1.output)
layer3.forward(layer2.output)
layer4.forward(layer3.output)

# Print predictions
print("PREDICTIONS")
print("New Input:")
print(new_input_data)
print("output (prediction):")
print(layer4.output)
