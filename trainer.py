import random
import numpy as np
from utils import evaluate
from kNNDTW import KnnDtw

class Trainer():

    def __init__(self, split_test=0.9, split_validation=0.7, seed=108, labels={1:'GOOD', 2:'BAD', 3:'WORST'}, data=None, data_labels=None):
        self.split_test = split_test
        self.split_validation = split_validation
        self.seed = seed
        self.labels = labels

        self.data = data
        self.data_labels = data_labels

        self._split()

    def _split(self):
        random.seed(self.seed)

        indices = np.arange(len(self.data))
        random.shuffle(indices)

        # Splitting into training and test

        split = int(round(len(self.data) * self.split_test)) # Split of 90%

        train_validation_index = indices[:split]
        test_index             = indices[split:]

        
        # Splitting into training and validation

        training_split = int(round(len(self.data) * self.split_validation)) # Split of 70%

        training_index   = train_validation_index[:training_split]
        validation_index = train_validation_index[training_split:]

        self.training_validation_data          = np.take(self.data, train_validation_index)
        self.training_validation_label_data    = np.take(self.data_labels, train_validation_index)        

        self.test_data          = np.take(self.data, test_index)
        self.test_label_data    = np.take(self.data_labels, test_index)


        self.training_data          = np.take(self.data, training_index)
        self.training_label_data    = np.take(self.data_labels, training_index)

        self.validation_data        = np.take(self.data, validation_index)
        self.validation_label_data  = np.take(self.data_labels, validation_index)


    def evaluate_model(self, k, max_warping_window, train_data, train_label, test_data, test_label):
        print '--------------------------'
        print '--------------------------\n'
        print 'Running for k = ', k
        print 'Running for w = ', max_warping_window
        
        model = KnnDtw(k_neighbours = k, max_warping_window = max_warping_window)
        model.fit(train_data, train_label)
        
        predicted_label, probability = model.predict(test_data)
        
        print '\nPredicted : ', predicted_label
        print 'Actual    : ', test_label
        
        accuracy, precision, recall, f1score = evaluate(self.labels, predicted_label, test_label)
        
        print 'Avg/Total Accuracy  :', accuracy
        print 'Avg/Total Precision :', precision
        print 'Avg/Total Recall    :', recall
        print 'Avg/Total F1 Score  :', f1score
        
        # result = np.zeros((len(ks),4))
        # result[0] = accuracy
        # result[1] = precision
        # result[2] = recall
        # result[3] = f1score


    def find_best_k(self, ks, max_warping_window):
        for index, k in enumerate(ks):
            self.evaluate_model(k, max_warping_window, self.training_data, self.training_label_data, self.validation_data, self.validation_label_data)

    def find_best_w(self, k, max_warping_windows):
        for index, w in enumerate(max_warping_windows):
            self.evaluate_model(k, w, self.training_data, self.training_label_data, self.validation_data, self.validation_label_data)

    def evalute_best_model(k, max_warping_window):
        self.evaluate_model(k, max_warping_window, self.training_validation_data, self.training_validation_label_data, self.test_data, self.test_label_data)


 
