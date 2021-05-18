# Logit Regression
# Have the function LogitRegression(arr) read the input array of 4 numbers x, y, a, b, separated by space, and return an output of two numbers for updated a and b (assume the learning rate is 1). Save up to 3 digits after the decimal points for a and b. The output should be a string in the format: a, b

# Logistic regression is a simple approach to do classification, and the same formula is also commonly used as the output layer in neural networks. We assume both the input and output variables are scalars, and the logistic regression can be written as:

# y = 1.0 / (1.0 + exp(-ax - b))

# After observing a data example (x, y), the parameter a and b can be updated using gradient descent with a learning rate.

import numpy as np

def LogitRegression(arr):
  learnrate = 1
  X = arr[0]
  y = arr[1]
  weights = arr[2]
  bias = arr[3]
  
  y_hat = 1/(1+np.exp(np.dot(X, -weights) - bias))
  new_weights = weights - learnrate * (y - y_hat) * X
  new_bias = bias - learnrate*(y - y_hat)
  return (round(new_weights,3), round(new_bias, 3))
