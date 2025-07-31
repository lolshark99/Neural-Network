import numpy as np
#this is a simple nn used to implement an XOR gate.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]).T  

Y = np.array([[0, 1, 1, 0]])  

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def deriv_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.max(0, z)

def parameters_init(input_size, hidden_size, output_size):
    params = {
        "W1": np.random.randn(hidden_size, input_size),
        "b1": np.zeros((hidden_size, 1)),
        "W2": np.random.randn(output_size, hidden_size),
        "b2": np.zeros((output_size, 1))
    }
    return params


def forward_prop(X, params):
    Z1 = np.dot(params["W1"], X) + params["b1"]
    A1 = sigmoid(Z1)
    Z2 = np.dot(params["W2"], A1) + params["b2"]
    A2 = sigmoid(Z2) 
    cache = (Z1, A1, Z2, A2)
    return A2, cache

def back_prop(X, Y, cache, params):
    m = X.shape[1]
    Z1, A1, Z2, A2 = cache

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(params["W2"].T, dZ2)
    dZ1 = dA1 * deriv_sigmoid(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def update_parameters(params, grads, learning_rate=0.1):
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params


params = parameters_init(2, 2, 1)# it  wil be a 2 layer nn with 2 nodes in 1st and 2nd layer and 1 node in  o/p layer

for i in range(10000):
    A2, cache = forward_prop(X, params)
    grads = back_prop(X, Y, cache, params)
    params = update_parameters(params, grads)

    if i % 1000 == 0:
        loss = np.mean((A2 - Y) ** 2)
        print(f"Iteration {i}, Loss: {loss}")

print("Predictions:")
print(A2.round())

