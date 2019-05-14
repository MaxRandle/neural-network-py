/*

general training algorithm in pseudocode

the .* operator represents element-wise multiplication of two matricies
the * operator represents matrix or scalar multiplication
w[n] represents the weight matrix at layer n
b[n] represents the bias matrix at layer n
a[n] represents the activation matrix at layer n
g(z) represents the activation function applied element-wise to the matrix z
x is input data
y is expected output
alpha is the learning rate

*/

//feedforward
//a[0] is the input matrix, a[n] is the output matrix
a[0] = x
for n in  (0 -> number_of_layers) {
    z = w[n] * a[n-1] + b[n]
    a[n] = g(z)
}


//backpropogation
//we do not wish to adjust layer 0 as this is the input layer
for n in (number_of_layers-1 -> 0) {
    if n === number_of_layers-1 {
        error[n] = y - a[n]
    } else{
        error[n] = w[n+1].T * error[n+1]
    }
    gradient[n] = g'(a[n]) .* error[n] * alpha
    delta[n] = gradient[n] * a[n-1].T
    w[n] += delta[n]
    b[n] += gradient[n]
}


/*
example use case

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

*/