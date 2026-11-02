"""Numpy logistic regression with 14 outputs, trained by full-batch gradient descent.
Rows are examples: X (m, n_x), Y and A (m, C), W (n_x, C), b (C,)."""

import numpy as np


def sigmoid(z):
    """1 / (1 + e^(-z)), cell by cell; any shape in, same shape out."""
    result = 1 / (1 + np.exp((-1) * z))

    return result


def initialize(n_x, n_classes):
    """Zero weights W (n_x, n_classes) and zero bias b (n_classes,).
    Zeros are fine: with no hidden layer every weight gets its own gradient."""
    W = np.zeros((n_x, n_classes))
    b = np.zeros(n_classes)

    return W, b


def propagate(W, b, X, Y):
    """One forward and backward pass over all m examples.
    Returns grads {"dW": (n_x, C), "db": (C,)} and the cost, the mean
    cross-entropy over all m x C cells. Gradients are divided by m only,
    so each category is trained like its own one-output model."""
    m = X.shape[0]
    Z = np.dot(X, W) + b
    A = sigmoid(Z)

    A_safe = np.clip(A, 1e-7, 1 - 1e-7)
    loss = -(Y * np.log(A_safe) + (1 - Y) * np.log(1 - A_safe))
    cost = np.mean(loss)

    dZ = A - Y
    dW = np.dot(X.transpose(), dZ) / m
    db = dZ.sum(axis=0) / m
    grads = {"dW": dW, "db": db}

    return grads, cost


def optimize(W, b, X, Y, learning_rate, num_iterations):
    """Full-batch gradient descent: num_iterations steps of W -= learning_rate * dW
    (same for b). Returns the trained W, b and the cost every 100 steps."""

    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(W, b, X, Y)
        W = W - learning_rate * grads["dW"]
        b = b - learning_rate * grads["db"]

        if i % 100 == 0:
            print(cost)
            costs.append(cost)

    return W, b, costs


def predict_proba(W, b, X):
    """Forward pass only: probabilities (m, C), no labels needed.
    The threshold is applied later, in classification_metrics."""
    prob = sigmoid(np.dot(X, W) + b)

    return prob
