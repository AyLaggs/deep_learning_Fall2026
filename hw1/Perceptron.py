# Given set of data points (and corresponding label), 
# this perceptron algorithm will go through each point in order, 
# and plot the hyperplane for every iteration. 
# It will then report the final converged solution (final weight and plot)

import torch
import matplotlib.pyplot as plt

def perceptron(X, y, num_iterations=1):
    """
    Perceptron algorithm for binary classification.

    Parameters:
    X : Matrix, shape (n_samples, n_features)
        Training data.
    y : vector, shape (n_samples,)
        Target values (labels).
    num_iterations : int, default=1
        Number of iterations to run the algorithm.

    Returns:
    weights : vector, shape (n_features,)
        Final weights after training.
    bias : float
        Final bias after training.
    """
    # Initialize weights and bias
    n_samples, n_features = X.shape
    weights = torch.zeros(n_features)
    bias = torch.tensor(0.0)

    # Store the history of weights and bias for convergence checking
    weights_history = []
    bias_history = []
    if_converge = False

    # Training loop
    for _ in range(num_iterations):
        for idx, x_i in enumerate(X):
            linear_output = torch.dot(x_i, weights) + bias
            y_predicted = torch.where(linear_output >= 0, torch.tensor(1.0), torch.tensor(-1.0))

            # Update weights and bias if prediction is wrong
            if y[idx] != y_predicted:
                weights += y[idx] * x_i
                bias += y[idx]

            plot_hyperplane(X, y, weights, bias)

        # Print the current weights and bias after each iteration
        print(f"Iteration {_ + 1}/{num_iterations}: Weights: {weights}, Bias: {bias}")

        # Store the current weights and bias for convergence checking
        weights_history.append(weights.clone())
        bias_history.append(bias.clone())
        
        # Print the change in weights and bias after each iteration
        if len(weights_history) > 1:
            print(f"Change in Weights: {weights - weights_history[-2]}, Change in Bias: {bias - bias_history[-2]}")
            print("-------------------------------------------------------------------------------------------------") 

        # If converged break out of the loop
        if len(weights_history) > 1 and torch.all(weights == weights_history[-2]) and bias == bias_history[-2]:
            if_converge = True
            break

    return weights, bias, if_converge

def plot_hyperplane(X, y, weights, bias):
    """
    Plots the data points and the decision boundary (hyperplane).

    Parameters:
    X : Matrix, shape (n_samples, n_features)
        Training data.
    y : vector, shape (n_samples,)
        Target values (labels).
    weights : vector, shape (n_features,)
        Current weights.
    bias : float
        Current bias.
    """
    plt.figure(figsize=(8, 6))
    
    # Plotting the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    
    # Plotting the decision boundary
    x1 = torch.linspace(X[:, 0].min()-2, X[:, 0].max()+2, 100)
    x2 = -(weights[0] * x1 + bias) / weights[1]
    plt.plot(x1.numpy(), x2.numpy(), 'k--', label='Decision Boundary')

    # Single weight vector on the decision boundary
    x1_start = torch.tensor(0.0)
    x2_start = -(weights[0] * x1_start + bias) / weights[1]
    plt.quiver(x1_start, x2_start, weights[0], weights[1], angles='xy', scale_units='xy', scale=1, color='g', label='Weight Vector')

    # Setting limits and labels
    plt.xlim(X[:, 0].min() - 2, X[:, 0].max() + 2)
    plt.ylim(X[:, 1].min() - 2, X[:, 1].max() + 2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.legend()
    plt.grid()
    plt.show()
    # plt.close()

# Example usage:
if __name__ == "__main__":    
    # Sample data points and labels
    # X = torch.tensor([[1, 1], [2, -2], [-1, -1.5], [-2, -1], [-2, 1], [1.5, -0.5], [1.5, 0.5]], dtype=torch.float32)
    # y = torch.tensor([1, -1, -1, -1, 1, 1, -1], dtype=torch.float32)

    # # OG
    # X = torch.tensor([[1, 1], [2, -2], [-1, -1.5], [-2, -1], [-2, 1], [1.5, -0.5]], dtype=torch.float32)
    # y = torch.tensor([1, -1, -1, -1, 1, 1], dtype=torch.float32)

    # Custom Dataset
    X = torch.tensor([[1, 1], [2, -2], [-1, -1.5], [-2, -1], [-2, 1], [1.5, -0.5], [-2.5, -0.5], [3, 3], [4, -4], [-3, -2], [-4, -3], [-3, 3], [3, -1]], dtype=torch.float32)
    y = torch.tensor([1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1], dtype=torch.float32)

    # Train the perceptron
    final_weights, final_bias, if_converge = perceptron(X, y, num_iterations=100)

    if if_converge:
        print("Converged!")
        print("Final Convergence Weights:", final_weights)
        print("Final Convergence Bias:", final_bias)
    else:
        print("Did not converge within the specified iterations.")
