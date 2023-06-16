__authors__ = '[1498566,1565408,1599349]'
__group__ = 'DL.10'

import numpy as np
import math
import operator
from pandas import unique
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxN matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #train_data.astype(float)
        #new_shape = train_data.shape[1] * train_data.shape[2]
        #self.train_data = np.reshape(train_data, (train_data.shape[0], new_shape))
        self.train_data = train_data.reshape(len(train_data), train_data[0].size).astype(float) # cambio para my_labeling

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        test_data = test_data.reshape(len(test_data), test_data[0].size).astype(float)
        self.neighbors = self.labels[np.argsort(cdist(test_data, self.train_data, 'cityblock'))[:, :k][:]]



    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        llista = []

        for n in self.neighbors:
            u,count = np.unique(n, return_counts=True)
            max = np.max(count)
            for c in n:
                if (np.count_nonzero(n == c) == max):
                    llista.append(c)
                    break

        return np.array(llista)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """

        #return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)

        self.get_k_neighbours(test_data, k)
        return self.get_class()