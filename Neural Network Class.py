#NN class by Max Randle

import numpy as np

class NN:
    def __init__(self, number_of_inputs):
        """number_of_inputs is the number of input neurons"""
        self.weights    = [np.identity(number_of_inputs)]
        self.biases     = [np.zeros((1, number_of_inputs))]
        self.numInp = number_of_inputs
        self.alpha = 0.1 #learning rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoidDerivative(self, z):
        return z * (1 - z)
    
    def g(self, z):
        return np.matrix(np.vectorize(self.sigmoid)(z))

    def dg(self, z):
        return np.matrix(np.vectorize(self.sigmoidDerivative)(z))

    def addDenseLayer(self, number_of_nodes):
        """adds a fully connected layer into the network. Initialized with random weights."""
        #Structure of each weight matrix is such that:
        #The number of ROWS is equal to the number of nodes in the PREVIOUS layer
        #The number of COLUMNS is equal to the number of nodes in the CURRENT layer

        self.weights    += [np.matrix(2 * np.random.random((self.weights[-1].shape[1], number_of_nodes)) - 1)]
        self.biases     += [np.matrix(np.ones((1, number_of_nodes)))]

    add = addDenseLayer

    def __repr__(self):
        out = "Layer View\n"
        for i in range(len(self.weights)):
            out += "Layer {} Weight ".format(i) + str(self.weights[i].shape) + "\n" + str(self.weights[i]) + "\nLayer {} Bias ".format(i) + str(self.biases[i].shape) + "\n" + str(self.biases[i]) + "\n"
        return out[:-1]

    def feedForward(self, data):
        """
        data must be a matrix  of shape (1, n) where n is the number of input values
        returns an array of activation matricies at each layer
        """
        ls = len(self.weights)
        a = [''] * ls
        a[0] = data
        for n in range(1, ls):
            a[n] = self.g(a[n-1] * self.weights[n] + self.biases[n])
        return a
    
    ff = feedForward

    def backPropogate(self, y, a):
        """
        y is the expected output matrix
        a is the list of activation matricies
        """
        error = y - a[-1]
        ls = len(self.weights)
        for n in range(1, ls)[::-1]:
            gradient = np.multiply(self.dg(a[n]), error) * self.alpha
            delta = gradient * self.weights[n].T
            self.weights[n] = self.weights[n] + delta
            self.biases[n] = self.biases[n] + gradient
            error =  error * self.weights[n].T
            
    bp = backPropogate

    def train(self, train_x, train_y, epochs):
        """
        train_x is a matrix of training data where each row is a set of training inputs.
        train_y is a matrix of training data where each row is the expected output of the correcponding inputs.
        epochs is the number of times the network should train over the given training data.
        """
        for n in range(epochs):
            activations = []
            for i in range(len(train_x)):
                activations = self.feedForward(train_x[i])
                self.backPropogate(train_y[i], activations)
                
            if (n == epochs//4) or (n == 2*epochs//4) or (n == 3*epochs//4) or (n == epochs-1):
            #if (n % epochs//4) == 0:
                print("Training loss: " + str(np.mean(np.abs(train_y[-1] - activations[-1]))))
                

"""

---Example Neural Network---

 L0                L1                              L2                              L3
         [        [b1]             ]     [        [b2]             ]     [        [b3]             ]
         [         +               ]     [         +               ]     [         +               ]
[a0]     [ [w1]   [z1]        [a1] ]     [ [w2]   [z2]        [a2] ]     [ [w3]   [z3]        [a3] ]
[a0]  *  [ [w1] * [z1] g(z)-> [a1] ]  *  [ [w2] * [z2] g(z)-> [a2] ]  *  [ [w3] * [z3] g(z)-> [a3] ]
[a0]     [ [w1]   [z1]        [a1] ]     [ [w2]   [z2]        [a2] ]     [ [w3]   [z3]        [a3] ]

---Example code for learning---

data set {
    [x] [y]
    [x] [y]
    [x] [y]
}

alpha = 0.1 //learning rate

func feedforward(x) {
    a = new array[4]
    a[0] = x
    for n in (1 -> 4) {
        a[n] = g(a[n-1] * w[n] + b[n])
    }
    return a
}

a = feedforward(x)

func backpropogate(a) {
    error = y - a[-1]
    for n in (3 -> 0) {
        gradient = g'(a[n]) .* error
        delta = gradient * w[n].T
        w[n] = w[n] + delta * alpha
        b[n] = b[n] + gradient * alpha
        error = error * w[n].T
    }
}

"""



#example training data to solve the XOR problem

train_x = np.matrix([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

train_y = np.matrix([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
])

x = train_x
y = train_y

nn = NN(2)
nn.add(2)
nn.add(2)

print(nn)

nn.train(x, y, 10000)

print(nn)







