#Neural Network class by Max Randle

import numpy as np

class NN:
    def __init__(self, number_of_inputs):
        self.activation = ["relu"]
        self.w = [np.matrix(np.identity(number_of_inputs))]
        self.b = [np.matrix(np.zeros((1, number_of_inputs)))]
        self.alpha = 0.05 #learning rate

    def sig(self, z):
        return np.matrix(np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(z))

    def sigDeriv(self, z):
        return np.matrix(np.vectorize(lambda x: x * (1-x))(z))

    def relu(self, z):
        return np.matrix(np.vectorize(lambda x: x if x > 0 else 0)(z))

    def reluDeriv(self, z):
        return np.matrix(np.vectorize(lambda x: 1 if x > 0 else 0)(z))

    def __repr__(self):
        out = "Layer View\n"
        for i in range(len(self.w)):
            out += "Layer {} Weight ".format(i) + str(self.w[i].shape) + "\n" + str(
                self.w[i]) + "\nLayer {} Bias ".format(i) + str(self.b[i].shape) + "\n" + str(self.b[i]) + "\n\n"
        return out[:-2]    

    def addDenseLayer(self, number_of_nodes, activation="relu"):
        """
        adds a fully connected layer into the network. Initialized with random weights.
        """
        if activation not in ["relu", "sigmoid"]:
            raise Exception("Activation function is not supported: {}".format(activation))
        self.activation += [activation]
        self.w += [np.matrix(2 * np.random.random((number_of_nodes, self.w[-1].shape[0])) - 1)]
        self.b += [np.matrix(2 * np.random.random((number_of_nodes, 1)) - 1)]

    add = addDenseLayer

    def feedForward(self, data):
        layers = len(self.w)
        a = [""] * layers
        a[0] = data
        for n in range(1, layers):
            z = self.w[n] * a[n-1] + self.b[n]
            if self.activation[n] == "relu":
                a[n] = self.relu(z)
            elif self.activation[n] == "sigmoid":
                a[n] = self.sig(z)             
        return a

    ff = feedForward

    def backPropogate(self, y, a):
        """
        y is the expected output
        a is the array of activation values from feed forward
        """
        layers = len(self.w)
        error = [""] * layers
        gradient = [""] * layers
        delta = [""] * layers

        for n in range(1, layers)[::-1]:
            #calculate error
            if n == (layers - 1):
                error[n] = y - a[-1] #if output layer
            else:
                error[n] = self.w[n+1].T * error[n+1]
            #calculate gradient
            if self.activation[n] == "relu":
                gradient[n] = np.multiply(self.reluDeriv(a[n]), error[n]) * self.alpha
            elif self.activation[n] == "sigmoid":
                gradient[n] = np.multiply(self.sigDeriv(a[n]), error[n]) * self.alpha
            #calculate delta
            delta[n] = gradient[n] * a[n-1].T
            #update weights/biases
            self.w[n] += delta[n]
            self.b[n] += gradient[n]
        
    def train(self, train_x, train_y, epochs):
        """
        train_x is a matrix of training data where each row is a set of training inputs.
        train_y is a matrix of training data where each row is the expected output of the correcponding inputs.
        epochs is the number of times the network should train over the given training data.
        """
        for n in range(epochs):
            y_error = [""] * len(train_x)
            activations = []
            for i in range(len(train_x)):
                activations = self.feedForward(train_x[i].T)
                y_error[i] = np.mean(np.abs(train_y[i].T - activations[-1]).T)
                self.backPropogate(train_y[i].T, activations)

            if (n % 100 == 0):
                print("Training loss: " + str(np.round(y_error, 5)))

# example training data to solve the XOR problem

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
nn.add(128)
nn.add(128)
nn.add(2, "sigmoid")

nn.train(x, y, 500 + 1)
