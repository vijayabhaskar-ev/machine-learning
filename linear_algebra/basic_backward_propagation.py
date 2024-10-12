import numpy as np
np.random.seed(42)
def backward_propagation(A, X, Y):
  
    m = X.shape[1]
 
    # Backward propagation: calculate dW, db.
    dZ = A - Y
    print("m:",X.T)
    dW = 1/m * np.matmul(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):

    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Retrieve each gradient from the dictionary "grads".
    dW = grads["dW"]
    db = grads["db"]
    
    # Update rule for each parameter.
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

def train_nn(parameters, A, X, Y):
    # Backpropagation. Inputs: "A, X, Y". Outputs: "grads".
    grads = backward_propagation(A, X, Y)
    
    # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
    parameters = update_parameters(parameters, grads)
    
    return parameters









# Number of examples
m = 5

# Number of input features
n_x = 3

# Number of output classes (assuming binary classification)
n_y = 1

# Create sample input data
X = np.random.randn(n_x, m)
print("X:",X)
# Create sample true labels (binary classification)
Y = np.random.randint(0, 2, (n_y, m))
print("Y:",Y)
# Create sample network output (predictions)
A = np.random.rand(1, m)
print("A:",A)
# Create sample parameters
parameters = {
    "W": np.random.randn(n_y, n_x),
    "b": np.random.randn(n_y, 1)
}
print("Parameters:",parameters)
# Now we can call the backward_propagation function
grads = train_nn(parameters, A, X, Y)


print("Gradients:",grads)
