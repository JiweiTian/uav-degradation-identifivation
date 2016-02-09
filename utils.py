import numpy as np
import csv
import sys

def load_labelled(csv_file_path):

    x_data = []
    label_data = []
    csv.field_size_limit(sys.maxsize)

    with open(csv_file_path, 'rU') as csvfile:
        uavreader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in uavreader:
            label_data.append(int(row[0]))
            x_data.append([float(ts) for ts in row[1].split()])

        # Convert to numpy for efficiency
        x_data     = np.array(x_data)
        label_data = np.array(label_data)

        return [x_data, label_data]

def load_test(csv_file_path):
    x_data = []
    csv.field_size_limit(sys.maxsize)

    with open(csv_file_path, 'rU') as csvfile:
        uavreader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in uavreader:
            x_data.append([float(ts) for ts in row[1].split()])

        # Convert to numpy for efficiency
        x_data = np.array(x_data)

        return x_data



def print_confusion_matrix(tp, fp, fn, tn):
        print '         | Predicted + | Predicted -'
        print 'Actual + |      '+str(tp)+'      |    ' + str(fn)
        print 'Actual - |      '+str(fp)+'      |    ' + str(tn)
        print ''

def evaluate(labels, label, test_label):
    accuracies = np.zeros(len(labels))
    precisions = np.zeros(len(labels))
    recalls = np.zeros(len(labels))
    f1scores = np.zeros(len(labels))

    for index,l in enumerate(labels):
        count = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for i in range(0,len(label)):
            if label[i] == l:
                count += 1
                if test_label[i] == label[i]:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                #if validation_label_data[i] == label[i]:
                if test_label[i] != l:
                    true_negative += 1
                else:
                    false_negative += 1

        print 'Label:', l
        print_confusion_matrix(true_positive, false_positive, true_negative, false_negative)

        acc = float(true_positive + true_negative)/len(label)

        precision = 0.0
        recall = 0.0
        f1score = 0.0
        
        if (true_positive + false_positive) > 0:
            precision = float(true_positive) / float(true_positive + false_positive)
        
        if (true_positive + false_negative) > 0:
            recall = float(true_positive)/float(true_positive + false_negative)
    
        if precision != 0 or recall != 0:
            f1score = float(2 * (precision * recall))/float(precision + recall)
    
        accuracies[index] = acc
        precisions[index] = precision
        recalls[index] = recall
        f1scores[index] = f1score

        print 'Accuracy:', acc
        print 'Precision:', precision
        print 'Recall:', recall
        print 'F1 Score', f1score
        # ...
        print '--------------'

    return [accuracies.mean(), precisions.mean(), recalls.mean(), f1scores.mean()]