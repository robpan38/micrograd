# MicroGrad: Building Neural Networks from Scratch

## Description

MicroGrad is a minimalist autograd engine and neural network library built from scratch in Python. This project demonstrates the fundamental concepts of automatic differentiation and neural networks by implementing them with clean, readable code. Inspired by PyTorch's approach, MicroGrad provides a clear view into how gradient-based machine learning works under the hood.

## Features

- **Custom Autograd Engine**: Built from first principles with a focus on clarity and educational value
- **Automatic Differentiation**: Computes gradients through computational graphs for backpropagation
- **Visualization Tools**: Graph visualization for computational operations and gradient flow
- **Neural Network Components**: Implementation of neurons, layers, and MLPs (Multi-Layer Perceptrons)
- **Training Framework**: Complete pipeline for training neural networks on sample data

## Technical Highlights

- Implementation of a Value class that tracks gradients through operations
- Support for basic operations (add, multiply, power, etc.) with automatic gradient tracking
- Tanh activation function with proper derivative calculations
- Topological sorting algorithm for efficient backpropagation
- Complete neural network architecture with modular design

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/micrograd.git](https://github.com/yourusername/micrograd.git)
   cd micrograd
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib graphviz
   ```

# Usage

The library can be used to create and train simple neural networks:
```python
# Create a simple MLP with 3 inputs, two hidden layers of 4 neurons, and 1 output
model = MLP(3, [4, 4, 1])

# Example training data
xs = [
    [2.0, 3.0, -1],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# Training loop
for k in range(100):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Backward pass
    loss.backward()
    
    # Update weights
    for p in model.parameters():
        p.data += -0.01 * p.grad
        p.grad = 0.0
```

# Visualization
MicroGrad includes tools to visualize the computational graph:
```python
# Create values and operations
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')

# Perform operations
n = x1*w1 + x2*w2 + b
o = n.tanh()

# Compute gradients
o.backward()

# Visualize the computational graph
draw_dot(o)
```

# Educational Value
Educational Value
This project is perfect for:
* Students learning about neural networks and backpropagation
* Developers wanting to understand the internals of deep learning frameworks
* Anyone interested in the mathematics behind   gradient-based optimization
