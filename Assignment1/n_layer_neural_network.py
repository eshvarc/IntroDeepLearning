from three_layer_neural_network import NeuralNetwork

__author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class DeepNetwork(NeuralNetwork):
    def __init__(self, num_layers, layer_dims, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param num_layers: the number of layers in the network including the input and output layers.
        :param layer_dims: a list of the number of units in each layer.
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        self.layers = [Layer(layer_dims[i], layer_dims[i + 1], actFun_type, reg_lambda, seed)
                       for i in range(self.num_layers - 1)]

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if type == 'tanh':
            return np.tanh(z)
        if type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        if type == 'relu':
            return np.where(z < 0, 0, z)

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if type == 'tanh':
            return 1 - np.square(np.tanh(z))
        if type == 'sigmoid':
            return np.exp(-z) / (np.square(1 + np.exp(-z)))
        if type == 'relu':
            return np.where(z < 0, 0, 1)

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''
        self.layers[0].feedforward(X, actFun)
        for i in range(1, self.num_layers - 1):
            self.layers[i].feedforward(self.layers[i - 1].a1, actFun)
        self.probs = self.layers[-1].feedforwardsoftmax(self.layers[-2].a1)

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        # y is a zero-one vector that represents the class of the output.
        # self.probs[np.arange(self.probs.shape[0], y]'s ith row contains the (i, y)th entry of self.probs.
        # So if the class of the ith data point is 0, then the expression contains (i, 0) in its ith row and if the
        # class of the ith data point is 1, then the expression contains (i,1) from self.probs in its ith row.

        # Then the log is taken to get the dersired log-loss for each data point. This is then summed over all data
        # points to get the complete loss.
        data_loss = -np.sum(np.log(self.probs[np.arange(self.probs.shape[0]), y]))

        # Add regulatization term to loss (optional)
        reg_sum = sum([np.sum(np.square(layer.W1)) for layer in self.layers])
        data_loss += self.reg_lambda / 2 * reg_sum
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''
        self.layers[-1].backpropsoftmax(self.layers[-1].cache, y)
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            layer.backprop((layer.cache, self.layers[i + 1].W1), self.layers[i + 1].dout)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            self.backprop(X, y)

            for layer in self.layers:
                layer.dW1 += self.reg_lambda * layer.W1

                # Gradient descent parameter update
                layer.W1 += -epsilon * layer.dW1
                layer.b1 += -epsilon * layer.db1

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)




class Layer(NeuralNetwork):
    def __init__(self, nn_first_dim, nn_second_dim, actFun_type, reg_lambda, seed):
        self.nn_first_dim = nn_first_dim
        self.nn_second_dim = nn_second_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_first_dim, self.nn_second_dim) / np.sqrt(self.nn_first_dim)
        self.b1 = np.zeros((1, self.nn_second_dim))

    def feedforward(self, input, actFun):
        # A @ B is the same as np.dot(A,B)
        self.z1 = (input @ self.W1) + self.b1
        # Acitvation function
        self.a1 = actFun(self.z1)
        self.cache = input

    def feedforwardsoftmax(self, input):
        self.z1 = (input @ self.W1) + self.b1
        self.cache = input
        self.probs = np.exp(self.z1 - np.max(self.z1, axis=1, keepdims=True))
        self.probs /= np.sum(self.probs, axis=1, keepdims=True)
        return self.probs

    def backprop(self, input, dout):
        dLda1 = dout @ input[1].T
        da1dz1 = self.diff_actFun(self.z1, self.actFun_type)
        self.dout = dLda1 * da1dz1

        self.dW1 = input[0].T @ self.dout
        self.db1 = np.sum(self.dout, axis=0)

    def backpropsoftmax(self, input, y):
        self.dout = self.probs.copy()
        N = input.shape[0]
        self.dout[np.arange(N), y] -= 1
        self.dout /= 1. * N

        self.dW1 = input.T @ self.dout
        self.db1 = np.sum(self.dout, axis=0)




def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()

    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    model = DeepNetwork(num_layers=3, layer_dims=[2, 100, 100, 2], actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()