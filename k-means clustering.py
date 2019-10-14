
import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import random


def findClosestCentroids(X, centroids):
    K = np.size(centroids, 0)
    indx = np.zeros((np.size(X, 0), 1))

    for i in range(0, np.size(X,0)):

        tempDist = np.sum(np.square(np.subtract(X[i, :], centroids[0, :])))
        tempIndex = 0

        for j in range(1, K):
            
            if np.sum(np.square(np.subtract(X[i, :], centroids[j, :]))) < tempDist:
                tempDist  = np.sum(np.square(np.subtract(X[i, :], centroids[j, :])))
                tempIndex = j

        indx[i] = tempIndex
    
    return indx


def computeCentroidMeans(X, indx, K):
    m = np.size(X, 0)
    n = np.size(X, 1)

    centroids = np.zeros((K, n))

    for i in range(0, K):
        counter = 0
        tempSum = np.zeros((1, n))

        for j in range(0, m):
            if indx[j] == i:
                counter = counter + 1
                tempSum[0, :] = tempSum[0, :] + X[j, :]
            
        centroids[i, :] = tempSum[0, :] / counter

    return centroids


def runKMeans(X, init_centroids, numIter):
    indx = np.zeros((np.size(X, 0), 1))
    K = np.size(init_centroids, 0)
    centroids = init_centroids

    for i in range(0, numIter):
        indx = findClosestCentroids(X, centroids)
        centroids = computeCentroidMeans(X, indx, K)

    return (centroids, indx)


def computeCost(X, centroids, indx):
    J = 0
    m = np.size(X, 0)
    K = np.size(centroids, 0)

    for i in range(0, K):
        for j in range(0, m):
            if indx[j] == i:
                J = J + np.sum(np.square(np.subtract(X[j, :], centroids[i, :])))

    return J / m


if __name__== "__main__":

    K = 3
    numIter = 10

    data = loadmat(os.path.join(sys.path[0], "ex7data2.mat"))

    X = np.array(data['X']) 

    # two sets of each variable. first to store current K-means run. second to store optimal result
    centroids = np.zeros((K, np.size(X, 1)))
    centroids_final = np.zeros((K, np.size(X, 1)))
    indx = np.zeros((np.size(X, 0), 1))
    indx_final = np.zeros((np.size(X, 0), 1))
    J = sys.maxsize
    J_final = sys.maxsize

    # run 20 times with random X values as init centroids to reduce risk of finding answer at local minimum
    for i in range(0, 20):
        cent1 = 0
        cent2 = 0
        cent3 = 0

        # randomize init centroids until they are not the same
        while cent1 == cent2 or cent1 == cent3 or cent2 == cent3:
            cent1 = random.randint(0, np.size(X, 0) - 1)
            cent2 = random.randint(0, np.size(X, 0) - 1)
            cent3 = random.randint(0, np.size(X, 0) - 1)

        centroids = np.array([X[cent1, :], X[cent2, :], X[cent3, :]])

        (centroids, indx) = runKMeans(X, centroids, numIter)
        J = computeCost(X, centroids, indx)

        if J < J_final:
            J_final = J
            centroids_final = centroids
            indx_final = indx


