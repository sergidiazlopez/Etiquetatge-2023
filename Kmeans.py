__authors__ = '1604158, 1599349, 1601959'
__group__ = 'PRACT2_1'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        X.astype(float)

        new_shape = X.shape[0] * X.shape[1]
        self.X = np.reshape(X, (new_shape, 3))

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'ICD'

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        cont = 0
        X = 0
        newPosition = 0
        trobat = False
        self.centroids = np.zeros([self.K, self.X.shape[1]])
        if self.options['km_init'].lower() == 'first':
            while (cont != self.K):
                for i in self.centroids:
                    if np.array_equal(i, self.X[X]):
                        trobat = True
                        break
                    else:
                        trobat = False

                if not trobat:
                    self.centroids[newPosition] = np.copy(self.X[X])
                    newPosition += 1
                    cont += 1
                X += 1
        elif self.options['km_init'].lower() == 'random':
            self.centroids = (np.random.rand(self.K, self.X.shape[1]))
            self.centroids = self.centroids.astype(float)
        elif self.options['km_init'].lower() == 'custom':
            self.centroids = np.zeros([self.K, self.X.shape[1]])
            self.centroids[0] = self.X[np.random.radint(self.X.shape([0])), :]
            for k in range(1, self.K):
                distancia = np.amin(distance(self.X, self.centroids), axis=1)
                self.centroids[k] = self.X[np.argmax(distancia), :]


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = np.array(self.centroids)
        novaCentroids = np.zeros([self.K, self.X.shape[1]])

        for i in range(self.K):
            if (self.labels == i).any():
                novaCentroids = self.X[self.labels == i]
                mitjana = np.mean(novaCentroids, axis=0)
                self.centroids[i] = np.copy(mitjana)


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        comparacion = self.centroids == self.old_centroids
        igual = comparacion.all()
        return igual


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()
        row = self.X.shape[0]
        col = self.X.shape[1]
        tam = row * col
        iguales = False
        i = 0
        while (i < tam) and (iguales == False):
            self.get_labels()
            self.get_centroids()
            if (self.converges() == True):
                iguales = True
            else:
                i += 1


    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        a = np.divide(1, len(self.X))
        b = np.sum(np.square(np.min(distance(self.X, self.centroids), axis=1)))
        return a * b

    def interClassDistance(self):
        sum = 0

        for i in range(len(self.centroids)):
            for j in range(i + 1, len(self.centroids)):
                sum += np.mean(np.square(self.centroids[i] - self.centroids[j]))

        return sum

    def fisherDistance(self):
        withinclass = self.withinClassDistance()
        interclass = self.interClassDistance()
        return withinclass / interclass


    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        fitting_function = self.withinClassDistance
        if self.options["fitting"] == "WCD":
            fitting_function = self.withinClassDistance
        elif self.options["fitting"] == "ICD":
            fitting_function = self.interClassDistance
        elif self.options["fitting"] == "Fisher":
            fitting_function = self.fisherDistance

        dist_previous = np.NaN
        for x in range(2, max_K):
            self.K = x
            self.fit()
            dist = fitting_function()
            if dist_previous and (1 - (dist / dist_previous) < 0.20):
                self.K = self.K - 1
                self.fit()
                return
            dist_previous = dist


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    K = C.shape[0]
    dist = np.zeros((X.shape[0], K))
    # suma = np.zeros(())
    for index in range(K):
        suma = ((X - C[index]) ** 2).sum(axis=1)
        dist[:, [index]] = np.sqrt(np.reshape(suma, (suma.shape[0], 1)))
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

        Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    llista = utils.get_color_prob(centroids)
    return utils.colors[np.argmax(llista, axis=1)]
