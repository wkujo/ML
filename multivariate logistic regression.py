
import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt


def sigmoid(X):
    return 1 / (1 + math.exp(-X))


# convert 2 first order features to all polynomial terms up to the degree-th power
def featureMapping(X, degree):

    result = np.ones((np.size(X, 0), 1))
    counter = 1

    for i in range(1, degree + 1):
        for j in range(0, i + 1):

            temp1 = np.power(X[:,0], (i - j)).reshape(np.size(data, 0), 1)
            temp2 = np.power(X[:,1], j).reshape(np.size(data, 0), 1)
            result = np.append(result, np.multiply(temp1, temp2), axis=1)
            
            counter = counter + 1
            # print("X1^" + str(i - j) + ", X2^" + str(j))

    return result


# compute the regularized cost
def computeCostGrad(X, Y, theta, lmbda):
    m = np.size(X, 0)
    n = np.size(X, 1)
    featureGrad = np.zeros((n, 1))

    sigmoidVector = np.vectorize(sigmoid)

    h = sigmoidVector(X.dot(theta))

    # compute cost    
    posTemp = np.multiply(-Y, np.log(h))
    negTemp = np.multiply(1 - Y, np.log(1 - h))
    
    error = np.subtract(posTemp, negTemp)
    regTerm = np.sum(np.square(theta[1:,:]))

    J = 1 / m * (np.sum(error) + regTerm * lmbda / 2)

    # compute gradients
    for i in range(0, n):
        featureGrad[i, 0] = np.sum(np.multiply(np.subtract(h, Y), X[:, i].reshape(np.size(data, 0), 1)))

        featureGrad[i, 0] = featureGrad[i, 0] / m

    # don't regularize intercept parameter
    for i in range(1, n):
        featureGrad[i, 0] = featureGrad[i, 0] + lmbda / m * theta[i, 0]

    return (J, featureGrad)


def gradDescent(X, Y, theta, lmbda, alpha, numIter):
    m = np.size(X, 0)
    n = np.size(X, 1)
    J_history = np.zeros((numIter, 1))    
    featureGrad = np.zeros((n, 1))

    for i in range(0, numIter):

        (J_history[i, 0], featureGrad) = computeCostGrad(X, Y, theta, lmbda)

        theta = theta - alpha / m * featureGrad

    return (J_history, theta)



if __name__ == "__main__":

    # [microchip parameter1, microchip parameter2, defective flag]
    data = np.array([[0.05127,0.69956,1.00000],
        [-0.09274,0.68494,1.00000],
        [-0.21371,0.69225,1.00000],
        [-0.37500,0.50219,1.00000],
        [-0.51325,0.46564,1.00000],
        [-0.52477,0.20980,1.00000],
        [-0.39804,0.03436,1.00000],
        [-0.30588,-0.19225,1.00000],
        [0.01671,-0.40424,1.00000],
        [0.13191,-0.51389,1.00000],
        [0.38537,-0.56506,1.00000],
        [0.52938,-0.52120,1.00000],
        [0.63882,-0.24342,1.00000],
        [0.73675,-0.18494,1.00000],
        [0.54666,0.48757,1.00000],
        [0.32200,0.58260,1.00000],
        [0.16647,0.53874,1.00000],
        [-0.04666,0.81652,1.00000],
        [-0.17339,0.69956,1.00000],
        [-0.47869,0.63377,1.00000],
        [-0.60541,0.59722,1.00000],
        [-0.62846,0.33406,1.00000],
        [-0.59389,0.00512,1.00000],
        [-0.42108,-0.27266,1.00000],
        [-0.11578,-0.39693,1.00000],
        [0.20104,-0.60161,1.00000],
        [0.46601,-0.53582,1.00000],
        [0.67339,-0.53582,1.00000],
        [-0.13882,0.54605,1.00000],
        [-0.29435,0.77997,1.00000],
        [-0.26555,0.96272,1.00000],
        [-0.16187,0.80190,1.00000],
        [-0.17339,0.64839,1.00000],
        [-0.28283,0.47295,1.00000],
        [-0.36348,0.31213,1.00000],
        [-0.30012,0.02705,1.00000],
        [-0.23675,-0.21418,1.00000],
        [-0.06394,-0.18494,1.00000],
        [0.06279,-0.16301,1.00000],
        [0.22984,-0.41155,1.00000],
        [0.29320,-0.22880,1.00000],
        [0.48329,-0.18494,1.00000],
        [0.64459,-0.14108,1.00000],
        [0.46025,0.01243,1.00000],
        [0.62730,0.15863,1.00000],
        [0.57546,0.26827,1.00000],
        [0.72523,0.44371,1.00000],
        [0.22408,0.52412,1.00000],
        [0.44297,0.67032,1.00000],
        [0.32200,0.69225,1.00000],
        [0.13767,0.57529,1.00000],
        [-0.00634,0.39985,1.00000],
        [-0.09274,0.55336,1.00000],
        [-0.20795,0.35599,1.00000],
        [-0.20795,0.17325,1.00000],
        [-0.43836,0.21711,1.00000],
        [-0.21947,-0.01681,1.00000],
        [-0.13882,-0.27266,1.00000],
        [0.18376,0.93348,0.00000],
        [0.22408,0.77997,0.00000],
        [0.29896,0.61915,0.00000],
        [0.50634,0.75804,0.00000],
        [0.61578,0.72880,0.00000],
        [0.60426,0.59722,0.00000],
        [0.76555,0.50219,0.00000],
        [0.92684,0.36330,0.00000],
        [0.82316,0.27558,0.00000],
        [0.96141,0.08553,0.00000],
        [0.93836,0.01243,0.00000],
        [0.86348,-0.08260,0.00000],
        [0.89804,-0.20687,0.00000],
        [0.85196,-0.36769,0.00000],
        [0.82892,-0.52120,0.00000],
        [0.79435,-0.55775,0.00000],
        [0.59274,-0.74050,0.00000],
        [0.51786,-0.59430,0.00000],
        [0.46601,-0.41886,0.00000],
        [0.35081,-0.57968,0.00000],
        [0.28744,-0.76974,0.00000],
        [0.08583,-0.75512,0.00000],
        [0.14919,-0.57968,0.00000],
        [-0.13306,-0.44810,0.00000],
        [-0.40956,-0.41155,0.00000],
        [-0.39228,-0.25804,0.00000],
        [-0.74366,-0.25804,0.00000],
        [-0.69758,0.04167,0.00000],
        [-0.75518,0.29020,0.00000],
        [-0.69758,0.68494,0.00000],
        [-0.40380,0.70687,0.00000],
        [-0.38076,0.91886,0.00000],
        [-0.50749,0.90424,0.00000],
        [-0.54781,0.70687,0.00000],
        [0.10311,0.77997,0.00000],
        [0.05703,0.91886,0.00000],
        [-0.10426,0.99196,0.00000],
        [-0.08122,1.10890,0.00000],
        [0.28744,1.08700,0.00000],
        [0.39689,0.82383,0.00000],
        [0.63882,0.88962,0.00000],
        [0.82316,0.66301,0.00000],
        [0.67339,0.64108,0.00000],
        [1.07090,0.10015,0.00000],
        [-0.04666,-0.57968,0.00000],
        [-0.23675,-0.63816,0.00000],
        [-0.15035,-0.36769,0.00000],
        [-0.49021,-0.30190,0.00000],
        [-0.46717,-0.13377,0.00000],
        [-0.28859,-0.06067,0.00000],
        [-0.61118,-0.06798,0.00000],
        [-0.66302,-0.21418,0.00000],
        [-0.59965,-0.41886,0.00000],
        [-0.72638,-0.08260,0.00000],
        [-0.83007,0.31213,0.00000],
        [-0.72062,0.53874,0.00000],
        [-0.59389,0.49488,0.00000],
        [-0.48445,0.99927,0.00000],
        [-0.00634,0.99927,0.00000],
        [0.63265,-0.03061,0.00000]])

    # split X and Y data
    Y = data[:,2].reshape(np.size(data, 0), 1)
    X = data[:, 0:2]

    X = featureMapping(X, 6)

    theta = np.zeros((np.size(X, 1), 1))
    
    lmbda = 1
    alpha = 10
    numIter = 2500

    # print(computeCostGrad(X, Y, theta, lmbda))

    (J_history, finalTheta) = gradDescent(X, Y, theta, lmbda, alpha, numIter)

    print(finalTheta)

    plt.plot(np.arange(numIter), J_history)
    plt.show()




# def featureRegularization(X):
#     paramInfo = np.zeros((len(X[0]), 2)) # [avg, std dev] for each parameter

#     for i in range(0, len(X[0])):
#         paramInfo[i,0] = np.average(X[:,i])
#         paramInfo[i,1] = np.std(X[:,i])

#     X_normal = np.array(X, dtype=float)
#     for i in range(0, len(X_normal)):
#         for j in range(0, len(X_normal[0])):
#             X_normal[i,j] = (X_normal[i,j] - paramInfo[j,0]) / paramInfo[j,1]
            
#     return X_normal


# # compute the non-regularized cost
# def computeCost(X, Y, theta):
#     m = np.size(X, 0)

#     h = X.dot(theta)
    
#     error = np.square(h - Y)
#     J = 1 / 2 / m * np.sum(error)
#     return J
    

# # iterate through and perform batch gradient descent
# def gradDescent(X, Y, theta, alpha, numIter):
#     m = np.size(X, 0)
#     n = np.size(X, 1)
#     J_history = np.zeros((numIter, 1))

#     featureGrad = np.zeros((np.size(X, 1), 1))

#     # iterate through
#     for i in range(0, numIter):

#         # for each iteration, calculate the partial derivative wrt each parameter
#         for j in range(0, n):  
#             if j == 0:
#                 featureGrad[j, 0] = np.sum(X.dot(theta) - Y)

#             else:
#                 tempX = X[:, j].reshape(np.size(X_normal, 0), 1)

#                 tempInnerGrad = np.multiply(X.dot(theta) - Y, tempX)
                
#                 featureGrad[j, 0] = np.sum(tempInnerGrad)
  
#         # update all parameters at the same time
#         theta = theta - alpha / m * featureGrad

#         J_history[i, 0] = computeCost(X, Y, theta)

#     return (theta, J_history)



# if __name__ == "__main__":

#     # sample data on housing
#     # [sq ft, num bedrooms, price]
#     data = np.array([[2104,3,399900],
#         [1600,3,329900],
#         [2400,3,369000],
#         [1416,2,232000],
#         [3000,4,539900],
#         [1985,4,299900],
#         [1534,3,314900],
#         [1427,3,198999],
#         [1380,3,212000],
#         [1494,3,242500],
#         [1940,4,239999],
#         [2000,3,347000],
#         [1890,3,329999],
#         [4478,5,699900],
#         [1268,3,259900],
#         [2300,4,449900],
#         [1320,2,299900],
#         [1236,3,199900],
#         [2609,4,499998],
#         [3031,4,599000],
#         [1767,3,252900],
#         [1888,2,255000],
#         [1604,3,242900],
#         [1962,4,259900],
#         [3890,3,573900],
#         [1100,3,249900],
#         [1458,3,464500],
#         [2526,3,469000],
#         [2200,3,475000],
#         [2637,3,299900],
#         [1839,2,349900],
#         [1000,1,169900],
#         [2040,4,314900],
#         [3137,3,579900],
#         [1811,4,285900],
#         [1437,3,249900],
#         [1239,3,229900],
#         [2132,4,345000],
#         [4215,4,549000],
#         [2162,4,287000],
#         [1664,2,368500],
#         [2238,3,329900],
#         [2567,4,314000],
#         [1200,3,299000],
#         [852,2,179900],
#         [1852,4,299900],
#         [1203,3,239500]])

    
#     # split X and Y data
#     Y = data[:,2].reshape(np.size(data, 0), 1)
#     X = data[:, 0:2]

#     X_normal = featureRegularization(X)

#     # add intercept feature
#     X_normal = np.append(np.ones((np.size(X_normal, 0), 1)), X_normal, axis=1)

#     alpha = 0.01
#     numIter = 400

#     # initialize feature parameters
#     theta = np.zeros((np.size(X_normal, 1), 1))

#     (theta, J_history) = gradDescent(X_normal, Y, theta, alpha, numIter)

#     # plt.plot(np.arange(numIter), J_history)
#     # plt.show()

#     print("Final equation (using normalized features): ")
#     print(str(theta[0, 0]) + " + " + str(theta[1, 0]) + " * (sq ft) + " + str(theta[2, 0]) + " * (num bedrooms) = price of house")