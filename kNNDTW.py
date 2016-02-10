import sys
import collections
import itertools

import numpy as np

from scipy.stats import mode

from dtw import dtw_distance


class KnnDtw(object):
    
    def __init__(self, k_neighbours=5, max_warping_window=10000):
        self.k_neighbours       = k_neighbours
        self.max_warping_window = max_warping_window
    
    # Public Methods

    def fit(self, x_training_data, x_labels):        
        self.x_training_data = x_training_data
        self.x_labels = x_labels
        
    def predict(self, x):
        
        distance_matrix = self._distance_matrix(x, self.x_training_data)
        
        # Retrieve the k nearest neighbours
        # distance_matrix.argsort()
        #       Sort the list distance_matrix and returns the sorted indices
        # [:, :self.k_neighbours]
        #       returns only the last k neighbours
        knn_indices = distance_matrix.argsort()[:, :self.k_neighbours]

        # Retrieve the k nearest labels with the indices
        knn_labels = self.x_labels[knn_indices]
        
        # Compute labels and probabilities using the mode (majority vote) ????
        mode_data           = mode(knn_labels, axis=1)

        result_label        = mode_data[0]
        result_probability  = mode_data[1] / self.k_neighbours

        # Return tuple. Ravel is a numpy function that flattens an array.
        # Doc: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ravel.html.
        return result_label.ravel(), result_probability.ravel()

    
    def _distance_matrix(self, x, y):
        count = 0

        x_shape = np.shape(x)
        y_shape = np.shape(y)

        distance_matrix         = np.zeros((x_shape[0], y_shape[0])) 
        distance_matrix_size    = x_shape[0] * y_shape[0]

        for i in xrange(0, x_shape[0]):
            for j in xrange(0, y_shape[0]):
                # Compute DTW
                distance_matrix[i, j] = dtw_distance(x[i], y[j], self.max_warping_window)

                # Update progress
                count += 1
                self._show_progress(distance_matrix_size, count)
    
        print '\r\n'

        return distance_matrix
    
    def _show_progress(self, n, i):
        print '\r%d/%d %f %%' % (i,n, (float(i)/float(n))*100.0),
        sys.stdout.flush()


