# Neural Network Implementation in Python

## Overview
This is a simple **fully connected feedforward neural network** implemented in Python without external libraries. It supports multiple hidden layers, uses the **ReLU activation function** for hidden layers, and **softmax** for the output layer. The network is trained using **backpropagation** and **gradient descent** with support for saving and loading model parameters.

## Features
- **Customizable Architecture**: Supports a variable number of hidden layers and neurons per layer.
- **He Initialization**: Initializes weights using He initialization for better training stability.
- **Activation Functions**:
  - **ReLU (Rectified Linear Unit)** for hidden layers
  - **Softmax** for multi-class classification
- **Backpropagation**:
  - Uses **gradient descent** for weight updates
  - Includes **gradient clipping** to prevent exploding gradients
- **Cross-Entropy Loss Function**: Optimized for multi-class classification.
- **Model Saving & Loading**: Saves trained weights and biases to a file using **pickle**.

## Class Breakdown

### `__init__` Method
Initializes the network parameters:
- `num_inputs`: Number of input neurons
- `num_hiddenLayers`: Number of hidden layers
- `num_hiddenLayer`: Number of neurons per hidden layer
- `num_outputs`: Number of output neurons
- `learning_rate`: Learning rate for gradient descent
- `save_model`: Boolean flag to save model weights and biases
- `load_model`: Boolean flag to load pre-trained model parameters

### Activation Functions
- **ReLU (`ReLU(x)`)**: Applies ReLU activation to input values.
- **ReLU Derivative (`ReLU_dr(x)`)**: Returns 1 where input is positive, 0 otherwise.
- **Softmax (`softmax(x)`)**: Converts output layer values into probabilities for classification.

### Forward Propagation (`forward(inputs)`)
Computes the network's output given an input.
- Passes inputs through each layer, applying activation functions.
- Returns the **softmax** probabilities of the output layer.

### Backpropagation (`backward(inputs, expected_outputs, predicted_outputs)`)
Performs weight and bias updates using the gradient descent algorithm.
- Computes the error between predicted and expected outputs using cross-entropy loss.
- Uses **ReLU derivative** for hidden layer weight updates.
- Applies **gradient clipping** to prevent exploding gradients.

### Loss Function (`cross_entropy_loss(predicted_outputs, expected_outputs)`)
Calculates the cross-entropy loss for classification.

### Training (`train(X, y, epochs)`)
Trains the neural network over a specified number of epochs.
- Calls `forward()` to get predictions.
- Calls `backward()` to update weights.
- Prints loss and accuracy every 10 epochs.
- Saves model weights if `save_model=True`.

### Usage Example:
In 

