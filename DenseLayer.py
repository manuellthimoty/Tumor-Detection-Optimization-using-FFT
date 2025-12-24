import numpy as np
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
    
    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias # w . x + b
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient