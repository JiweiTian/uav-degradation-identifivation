import sys
import collections
import itertools

import numpy as np

from scipy.stats import mode


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

    # Private Methods

    def _distance(self, x, y):
        return abs( x - y )

    def _dtw_distance(self, timeserie_a, timeserie_b):

        # Initialising variables with numpy structures
        timeserie_a = np.array(timeserie_a)
        timeserie_b = np.array(timeserie_b)

        M = len(timeserie_a)
        N = len(timeserie_b)

        # Create cost matrix by filling it with very large integers, because the dynamic programming functions
        # uses the min(...) function. Thus, it is better then initializing with 0 or None.
        cost_matrix = sys.maxint * np.ones( (M, N) )

        # Initialising the first cell
        cost_matrix[0, 0] = self._distance(timeserie_a[0], timeserie_b[0])

        # Initializing the first row
        for i in xrange(1, M):
            cost_matrix[i, 0] = cost_matrix[i-1, 0] + self._distance(timeserie_a[i], timeserie_b[0])

        # Initializing the first column
        for i in xrange(1, N):
            cost_matrix[0, i] = cost_matrix[0, i-1] + self._distance(timeserie_a[0], timeserie_b[i])

        # Run trough the reste of the cost_matrix and stay withing the limits of the
        # warping window. Performs the "main dynamic programming function".
        for i in xrange(1, M):
            from_max_warping_window = max(1, i - self.max_warping_window)
            to_max_warping_window   = min(N, i + self.max_warping_window)

            for j in xrange(from_max_warping_window, to_max_warping_window):
                choice_1 = cost_matrix[i - 1, j - 1]
                choice_2 = cost_matrix[i - 1, j    ]
                choice_3 = cost_matrix[i    , j - 1]

                cost_min_choices = min(choice_1, choice_2, choice_3)

                cost_matrix[i, j] = cost_min_choices + self._distance(timeserie_a[i], timeserie_b[j])

        # Return the DTW distance ([-1,-1] return the last item)
        return cost_matrix[-1, -1]
    
    def _distance_matrix(self, x, y):
        count = 0

        x_shape = np.shape(x)
        y_shape = np.shape(y)

        distance_matrix         = np.zeros((x_shape[0], y_shape[0])) 
        distance_matrix_size    = x_shape[0] * y_shape[0]

        for i in xrange(0, x_shape[0]):
            for j in xrange(0, y_shape[0]):
                # Compute DTW
                distance_matrix[i, j] = self._dtw_distance(x[i], y[j])

                # Update progress
                count += 1
                self._show_progress(distance_matrix_size, count)
    
        print '\r\n'

        return distance_matrix
    
    def _show_progress(self, n, i):
        print '\r%d/%d %f %%' % (i,n, (float(i)/float(n))*100.0),
        sys.stdout.flush()


