from numpy import random, dot, array, exp


# define sigmoid function
def sigmoid(x):
    return 1 / (1 + exp(-x))


# derivative of sigmoid function to find local minima of loss function
def sigmoid_derivative(x):
    return x * (1 - x)


# adjust weights on each iteration
def minimize(training_inputs, z, a, output, weights_hidden, weights_output, error):
    adjustment_input = dot(training_inputs.T, error * sigmoid_derivative(output))
    adjustment_hidden = dot(z, error * sigmoid_derivative(output))
    adjustment_output = dot(a, error * sigmoid_derivative(output))
    weights_hidden[:3] +=  adjustment_input
    weights_hidden[3:] += adjustment_hidden
    weights_output += adjustment_output
    return weights_hidden, weights_output


# feedforward and backpropagate on each iteration for 10k iterations
def train(training_inputs, training_outputs, weights_hidden, weights_output):
    for i in range(10000):
        # z = summation(xi.wi) + bias (bias not considered here)
        # output = sigmoid(z)
        # first layer
        z1 = sigmoid(dot(training_inputs, weights_hidden[0]))
        z2 = sigmoid(dot(training_inputs, weights_hidden[1]))
        z3 = sigmoid(dot(training_inputs, weights_hidden[2]))
        
        # second layer
        a1 = sigmoid(dot(array([z1, z2, z3]).T, weights_hidden[3]))
        a2 = sigmoid(dot(array([z1, z2, z3]).T, weights_hidden[4]))
        a3 = sigmoid(dot(array([z1, z2, z3]).T, weights_hidden[5]))
        
        # output layer
        output = sigmoid(dot(array([a1, a2, a3]).T, weights_output))
        
        # error = actual - predicted
        error = training_outputs - output
        
        # new weights
        weights_hidden, weights_output = minimize(training_inputs, array([z1, z2, z3]), array([a1, a2, a3]), output, weights_hidden, weights_output, error)

    return weights_hidden, weights_output


# predict for custom inputs
def predict(inputs, weights_hidden, weights_output):
    z1 = sigmoid(dot(inputs, weights_hidden[0]))
    z2 = sigmoid(dot(inputs, weights_hidden[1]))
    z3 = sigmoid(dot(inputs, weights_hidden[2]))

    a1 = sigmoid(dot(array([z1, z2, z3]).T, weights_hidden[3]))
    a2 = sigmoid(dot(array([z1, z2, z3]).T, weights_hidden[4]))
    a3 = sigmoid(dot(array([z1, z2, z3]).T, weights_hidden[5]))

    return sigmoid(dot(array([a1, a2, a3]).T, weights_output))


if __name__ == '__main__':
    # init training and testing data
    training_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = array([[0, 1, 1, 0]]).T
    
    # so that every time, the same random numbers are generated
    random.seed(1)
    
    # initialise random weights
    # weights_hidden => weights for input and hidden layers
    # weights_ouptut => weights for output layer
    weights_hidden = 2 * random.random((6, 3)) - 1
    weights_output = 2 * random.random((3, 1)) - 1
    
    # get final weights after training
    weights_hidden, weights_output = train(training_inputs, training_outputs, weights_hidden, weights_output)
    
    # predict for custom inputs
    output = predict(array(list(map(int, input().split()))), weights_hidden, weights_output)

    print(float(output))