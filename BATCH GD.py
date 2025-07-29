from sklearn.datasets import fetch_california_housing
import numpy as np

data = fetch_california_housing(as_frame=True)
X = data.data.to_numpy()     
Y = data.target.to_numpy()   


n1 = int(input("No of nods in 1st layer"))
n2 = int(input("No of nods in 2nd layer"))
n3 = 1

def tanh(z):
    return np.tanh(z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def deriv_sigmoid(z):
    return sigmoid(z) - (1 - sigmoid(z))

def tanh_deriv(z):
    return 1 - np.tanh(z)**2


def params_init(n1 , n2 , n3):
    parameters = {
        "W1":np.random.randn(n1 , 2),
        "b1":np.zeros((n1 , 1)),
        "W2":np.random.randn(n2 , n1),
        "b2":np.zeros((n2 ,1)),
        "W3":np.random.randn(n3 , n2),
        "b3":np.zeros((n3 , 1))
    }
    return parameters

def forward_prop(parameters):
    Z1 = np.dot(parameters["W1"] , X) + parameters["b1"]
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1 , parameters["W2"]) + parameters["b2"]
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2 , parameters["W3"]) + parameters["b3"]
    output = tanh(Z3)
    cache = (Z1 , A1 , Z2 , A2 , Z3)    
    return output , cache

def back_prop(X , Y , parameters , cache , output):
    m = X.shape[0]
    Z1 , A1 , Z2 , A2 , Z3 = cache

    dz3 = output - Y
    dw3 = (1 / m) * np.dot(dz3 , A2)
    db3 = (1 / m) * np.sum(dz3)
    dA2 = np.dot(dz3 , dw3.T)
    #grads from the 3rd layer 

    dz2 = deriv_sigmoid(Z2) * dA2
    dw2 = (1/m) * np.dot(dz2 , A1)
    db2 = (1/m) * np.sum(dz2)
    dA1 = np.dot(dz2 , dw2.T)

    #grads from 2nd layer

    dz1 = deriv_sigmoid(Z1) * dA1
    dw1 = (1/m) * np.dot(dz1 , X.T)
    db1 = (1/m) * np.sum(dz1)

    grads = {
        "dW1": dw1, "db1": db1,
        "dW2": dw2, "db2": db2,
        "dW3": dw3, "db3": db3
    }

    return grads
def update_parameters(grads , parameters):
    parameters["W1"] -= 0.01 * grads["dW1"]
    parameters["b1"] -= 0.01 * grads["db1"]
    parameters["W2"] -= 0.01 * grads["dW2"]
    parameters["b2"] -= 0.01 * grads["db2"]
    parameters["W3"] -= 0.01 * grads["dW3"]
    parameters["b3"] -= 0.01 * grads["db3"]
    return parameters

parameters = params_init(n1 , n2 , n3)

for i in range(1000):
    
    output , cache = forward_prop(parameters)
    grads = back_prop(X , Y , parameters , cache , output)
    parameters = update_parameters(grads , parameters)





