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
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

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

        # YOU IMPLEMENT YOUR feedforward HERE
        # A @ B is the same as np.dot(A,B)
        self.z1 = (X @ self.W1) + self.b1
        # Acitvation function
        self.a1 = actFun(self.z1)
        # Affine transform with weights and bias obtained from joining hidden and final layer.
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Softmax function - computing the numerator. We subtract by the maximum component out of each component in the
        # z2 vector. This allows numerical stability while maintaining the probabilities.
        self.probs = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        # Divide by the denominator i.e. the sum of all the individual numerators (components => axis 1).
        self.probs /= np.sum(self.probs, axis=1, keepdims=True)

        # Side note: The keepdims=True flag keeps the resultant in the right shape for broadcasting. Eg: The shape of
        # self.z2 is (200, 2). The np.max(z2, axis=1) operator would return a vector (200,). This vector cannot be
        # directly subtracted. However, if the np.max(z2, axis=1, keepdims=True) operator is used, the result is a
        # (200, 1) "matrix" that can be broadcasted (and therefore directly subtracted).

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
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
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

        # IMPLEMENT YOUR BACKPROP HERE
        dLdz2 = self.probs.copy()
        N = X.shape[0]
        dLdz2[np.arange(N), y] -= 1
        dLdz2 /= 1. * N

        dW2 = self.a1.T @ dLdz2

        db2 = np.sum(dLdz2, axis=0)

        dLda1 = dLdz2 @ self.W2.T

        da1dz1 = self.diff_actFun(self.z1, self.actFun_type)
        dLdz1 = dLda1 * da1dz1

        dW1 = X.T @ dLdz1

        db1 = np.sum(dLdz1, axis=0)

        # print(f"db1 {db1}, db2 {db2}")
        return dW1, dW2, db1, db2

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
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

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

def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()

    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, actFun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type='sigmoid')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type='relu')

    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=200, nn_output_dim=2, actFun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=200, nn_output_dim=2, actFun_type='sigmoid')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=200, nn_output_dim=2, actFun_type='relu')
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()