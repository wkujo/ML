
import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

from sklearn.svm import SVC


if __name__ == "__main__":

    data = loadmat(os.path.join(sys.path[0], "ex6data2.mat"))

    # split X and Y data
    Y = np.array(data['y']) 
    X = np.array(data['X']) 

    X_group1 = np.array([X[i] for i in range(0, np.size(X, 0)) if Y[i] == 0])
    X_group2 = np.array([X[i] for i in range(0, np.size(X, 0)) if Y[i] == 1])

    # visualize the two subgroups
    # plt.plot(X_group1[:,0], X_group1[:,1], 'r+', marker='o')
    # plt.plot(X_group2[:,0], X_group2[:,1], 'r+', marker='+')
    # plt.show()

    C_input = 100
    gamma_input = 10

    # using no kernel / linear kernel
    # svclassifer = SVC(kernel='linear', C=C_input)

    # using gaussian kernel
    svclassifer = SVC(kernel='rbf', C=C_input, gamma=gamma_input)
    svclassifer.fit(X, Y.ravel())

    h = svclassifer.predict(X)
   
    plt.plot(X_group1[:,0], X_group1[:,1], 'r+', marker='o')
    plt.plot(X_group2[:,0], X_group2[:,1], 'r+', marker='+')
    
    axes = plt.gca()
    x1lim = axes.get_xlim()
    x2lim = axes.get_ylim()

    xx1 = np.linspace(x1lim[0], x1lim[1])
    xx2 = np.linspace(x2lim[0], x2lim[1])
    XX2, XX1 = np.meshgrid(xx2, xx1)
    xy = np.vstack([XX1.ravel(), XX2.ravel()]).T
    Z = svclassifer.decision_function(xy).reshape(XX1.shape)

    # plot decision boundary and margins
    plt.contour(XX1, XX2, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    plt.show()