import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):  # lr being the learning rate and n_iters being the numberof iterations
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None  # a
        self.bias = None  # b

    def fit(self, X, Y):  # takes the training samples and the labels and it is the training step and gradient descent
        # init param
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) #for each compononent we put in a zero
        self.bias = 0

        #the gradient descent
        for _ in range (self.n_iters):
            predicted_y = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (predicted_y - Y))
            db = (1/n_samples) * np.sum(predicted_y-Y)

            self.weights -= self.lr * dw
            self.bias -= self.lr *db


    def predict(self,X):  # takes test samples and approximates the label and return it
        pass