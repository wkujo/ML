
import numpy as np
import pandas as pd
import statistics
import sys
import matplotlib.pyplot as plt


def featureRegularization(X):
    paramInfo = np.zeros((len(X[0]), 2)) # [avg, std dev] for each parameter

    for i in range(0, len(X[0])):
        paramInfo[i,0] = np.average(X[:,i])
        paramInfo[i,1] = np.std(X[:,i])

    X_normal = np.array(X, dtype=float)
    for i in range(0, len(X_normal)):
        for j in range(0, len(X_normal[0])):
            X_normal[i,j] = (X_normal[i,j] - paramInfo[j,0]) / paramInfo[j,1]
            
    return X_normal


# compute the non-regularized cost
def computeCost(X, Y, theta):
    m = np.size(X, 0)

    h = X.dot(theta)
    
    error = np.square(h - Y)
    J = 1 / 2 / m * np.sum(error)
    return J
    

# iterate through and perform batch gradient descent
def gradDescent(X, Y, theta, alpha, numIter):
    m = np.size(X, 0)
    n = np.size(X, 1)
    J_history = np.zeros((numIter, 1))

    featureGrad = np.zeros((np.size(X, 1), 1))

    # iterate through
    for i in range(0, numIter):

        # for each iteration, calculate the partial derivative wrt each parameter
        for j in range(0, n):  
            if j == 0:
                featureGrad[j, 0] = np.sum(X.dot(theta) - Y)

            else:
                tempX = X[:, j].reshape(np.size(X_normal, 0), 1)

                tempInnerGrad = np.multiply(X.dot(theta) - Y, tempX)
                
                featureGrad[j, 0] = np.sum(tempInnerGrad)
  
        # update all parameters at the same time
        theta = theta - alpha / m * featureGrad

        J_history[i, 0] = computeCost(X, Y, theta)

    return (theta, J_history)



if __name__ == "__main__":

    # sample data on housing
    # [sq ft, num bedrooms, price]
    data = np.array([[2104,3,399900],
        [1600,3,329900],
        [2400,3,369000],
        [1416,2,232000],
        [3000,4,539900],
        [1985,4,299900],
        [1534,3,314900],
        [1427,3,198999],
        [1380,3,212000],
        [1494,3,242500],
        [1940,4,239999],
        [2000,3,347000],
        [1890,3,329999],
        [4478,5,699900],
        [1268,3,259900],
        [2300,4,449900],
        [1320,2,299900],
        [1236,3,199900],
        [2609,4,499998],
        [3031,4,599000],
        [1767,3,252900],
        [1888,2,255000],
        [1604,3,242900],
        [1962,4,259900],
        [3890,3,573900],
        [1100,3,249900],
        [1458,3,464500],
        [2526,3,469000],
        [2200,3,475000],
        [2637,3,299900],
        [1839,2,349900],
        [1000,1,169900],
        [2040,4,314900],
        [3137,3,579900],
        [1811,4,285900],
        [1437,3,249900],
        [1239,3,229900],
        [2132,4,345000],
        [4215,4,549000],
        [2162,4,287000],
        [1664,2,368500],
        [2238,3,329900],
        [2567,4,314000],
        [1200,3,299000],
        [852,2,179900],
        [1852,4,299900],
        [1203,3,239500]])

    
    # split X and Y data
    Y = data[:,2].reshape(np.size(data, 0), 1)
    X = data[:, 0:2]

    X_normal = featureRegularization(X)

    # add intercept feature
    X_normal = np.append(np.ones((np.size(X_normal, 0), 1)), X_normal, axis=1)

    alpha = 0.01
    numIter = 400

    # initialize feature parameters
    theta = np.zeros((np.size(X_normal, 1), 1))

    (theta, J_history) = gradDescent(X_normal, Y, theta, alpha, numIter)

    # plt.plot(np.arange(numIter), J_history)
    # plt.show()

    print("Final equation (using normalized features): ")
    print(str(theta[0, 0]) + " + " + str(theta[1, 0]) + " * (sq ft) + " + str(theta[2, 0]) + " * (num bedrooms) = price of house")