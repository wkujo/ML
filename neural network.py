
import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

from PIL import Image


def sigmoid(X):
    return 1 / (1 + math.exp(-X))


def sigmoidGradient(X):
    return sigmoid(X) * (1 - sigmoid(X))


# display 20 random examples
def displayImage(X):
    fig = plt.figure(figsize=(10,10))
    rows = 4
    columns = 5

    for i in range(1, rows * columns + 1):
        nextEx = np.random.randint(1, 5000)
        img = np.reshape(X[nextEx],(20,20))
        fig.add_subplot(rows, columns, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap="gray")
    
    plt.show()


def nnCostFunction(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    m = np.size(X, 0)
    J = 0
    sigmoid_v = np.vectorize(sigmoid)

    # forward prop
    a1 = np.append(np.ones((np.size(X, 0), 1)), X, axis=1)
    z2 = a1.dot(np.transpose(theta1))
    g2 = sigmoid_v(z2)
    a2 = np.append(np.ones((np.size(g2, 0), 1)), g2, axis=1)
    z3 = a2.dot(np.transpose(theta2))
    h = sigmoid_v(z3)

    y_output = np.zeros((np.size(X, 0), num_labels))
    # data originally created to use for 1 based index languages, not 0 based index
    for i in range(0, np.size(X,0)):
        y_output[i, y[i] - 1] = 1

    # compute cost    
    posTemp = np.multiply(-y_output, np.log(h))
    negTemp = np.multiply(np.subtract(1, y_output), np.log(np.subtract(1, h)))

    error = np.subtract(posTemp, negTemp)
    
    # don't apply regularization to constant's thetas
    regTerm = np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:]))

    J = 1 / m * (np.sum(error) + regTerm * lmbda / 2)

    # back prop
    sigmoidGradient_v = np.vectorize(sigmoidGradient)
    d3 = np.zeros((num_labels, 1))
    d2 = np.zeros((np.size(theta2, 1), 1))
    delta1 = np.zeros((np.size(theta1, 0), np.size(theta1, 1)))
    delta2 = np.zeros((np.size(theta2, 0), np.size(theta2, 1)))

    for i in range(0, m):
        d3 = np.subtract(h[i,:], y_output[i,:]).reshape(np.size(h, 1), 1)
  
        temp = np.append(np.ones((np.size(z2, 0), 1)), z2, axis=1)
  
        d2 = np.transpose(theta2).dot(np.transpose(d3).reshape(np.size(d3, 0), 1))
        d2 = np.multiply(d2, sigmoidGradient_v(temp[i, :]).reshape(np.size(temp, 1), 1))

        # drop value associated with added constant
        d2 = d2[1:]

        delta2 = delta2 + d3.dot(a2[i,:].reshape(1, np.size(a2, 1)))
        delta1 = delta1 + d2.dot(a1[i,:].reshape(1, np.size(a1, 1)))

    theta1_reg_term = theta1
    theta1_reg_term[:, 1] = 0

    theta2_reg_term = theta2
    theta2_reg_term[:, 1] = 0


    theta1_grad = delta1 / m + lmbda * theta1_reg_term / m
    theta2_grad = delta2 / m + lmbda * theta2_reg_term / m
    
    return (J, theta1_grad, theta2_grad)



def gradDescent(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, alpha, numIter):
    m = np.size(X, 0)
    J_history = np.zeros((numIter, 1))

    for i in range(0, numIter):

        (J_history[i, 0], theta1_grad, theta2_grad) = nnCostFunction(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
        
        theta1 = theta1 - alpha / m * theta1_grad
        theta2 = theta2 - alpha / m * theta2_grad

    return (J_history, theta1, theta2)



if __name__ == "__main__":

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    data = loadmat(os.path.join(sys.path[0], "ex4data1.mat"))

    # split X and Y data
    Y = np.array(data['y']) # 5000x1
    X = np.array(data['X']) # 5000x400

    # see 20 examples of what we are working with
    # displayImage(X)

    # dataTheta = loadmat(os.path.join(sys.path[0], "ex4weights.mat"))
    # theta1 = np.array(dataTheta['Theta1'])
    # theta2 = np.array(dataTheta['Theta2'])    

    # if not loading init thetas
    theta1 = np.random.rand(25, 401)
    theta2 = np.random.rand(10, 26)

    lmbda = 1
    alpha = 10
    numIter = 1000

    (J_history, theta1, theta2) = gradDescent(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, X, Y, lmbda, alpha, numIter)

    plt.plot(np.arange(numIter), J_history)
    plt.show()





