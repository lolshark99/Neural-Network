from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

X_axis = []
Y_axis = []



learning_rate = 0.1
def tanh(z): return np.tanh(z)
def tanh_deriv(z): return 1 - np.tanh(z)**2
def sigmoid(z): return 1 / (1 + np.exp(-z))
def deriv_sigmoid(z): s = sigmoid(z); return s * (1 - s)

# Load and preprocess data
data = load_iris()
X = data.data                      # (150, 4)
y = data.target.reshape(-1, 1)    # (150, 1)

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y).T    # (3, 150)

# Hyperparameters
n1, n2, n3 = 10, 5, 3
epochs = 1000
learning_rate = 0.1
batch_size = 5
m = X.shape[0]

# Parameter initialization
def parameters_init():
    return {
        "W1": np.random.randn(n1, 4),
        "b1": np.zeros((n1, 1)),
        "W2": np.random.randn(n2, n1),
        "b2": np.zeros((n2, 1)),
        "W3": np.random.randn(n3, n2),
        "b3": np.zeros((n3, 1)),
    }

# Forward pass
def forward_prop(X, parameters):
    Z1 = parameters["W1"] @ X + parameters["b1"]     # (10, batch)
    A1 = sigmoid(Z1)                                 # (10, batch)
    Z2 = parameters["W2"] @ A1 + parameters["b2"]     # (5, batch)
    A2 = tanh(Z2)                                     # (5, batch)
    Z3 = parameters["W3"] @ A2 + parameters["b3"]     # (3, batch)
    A3 = tanh(Z3)                                     # (3, batch)
    return (Z1, A1, Z2, A2, Z3, A3)

# Backward pass
def back_prop(X, Y, parameters, cache):
    m = X.shape[1]
    Z1, A1, Z2, A2, Z3, A3 = cache

    dZ3 = A3 - Y                                       # (3, m)
    dW3 = (1/m) * dZ3 @ A2.T                           # (3, 5)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)  # (3, 1)

    dA2 = parameters["W3"].T @ dZ3                     # (5, m)
    dZ2 = dA2 * tanh_deriv(Z2)                         # (5, m)
    dW2 = (1/m) * dZ2 @ A1.T                           # (5, 10)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # (5, 1)

    dA1 = parameters["W2"].T @ dZ2                     # (10, m)
    dZ1 = dA1 * deriv_sigmoid(Z1)                      # (10, m)
    dW1 = (1/m) * dZ1 @ X.T                            # (10, 4)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)  # (10, 1)

    return {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3
    }

# Parameter update
def update_parameters(grads , parameters):
    parameters["W1"] -= grads["dW1"] * learning_rate
    parameters["b1"] -= grads["db1"] * learning_rate
    parameters["W2"] -= grads["dW2"] * learning_rate
    parameters["b2"] -= grads["db2"] * learning_rate
    parameters["W3"] -= grads["dW3"] * learning_rate
    parameters["b3"] -= grads["db3"] * learning_rate
    return parameters


# Training loop
parameters = parameters_init()
for epoch in range(epochs):
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[:, permutation]
    total = 0
    X_axis.append(epoch)

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i+batch_size].T     # (4, batch)
        Y_batch = Y_shuffled[:, i:i+batch_size]     # (3, batch)

        cache = forward_prop(X_batch, parameters)
        grads = back_prop(X_batch, Y_batch, parameters, cache)
        parameters = update_parameters(grads, parameters)

    _, _, _, _, _, A3_full = forward_prop(X.T, parameters)
    loss = np.mean((A3_full - Y)**2)
    Y_axis.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

plt.plot(X_axis , Y_axis)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.grid(True)