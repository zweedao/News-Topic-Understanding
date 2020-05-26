""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import propername
import util
import sys
import csv
import numpy as np
from collections import defaultdict
from scipy.optimize import fmin_l_bfgs_b

class MaximumEntropyModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self, batch_size=None, pgtol=0.005, w_scale=0.1):
        self.batch_size = batch_size #mini-batch size for gradient descent
        self.pgtol = pgtol #early stopping criteria for training
        self.w_scale = w_scale #factor to scale initial random normal w values

    def objective_function(self, w):

        dot_products = []
        for i in range(self.num_labels):
            w_subset = w[i*self.num_features:(i+1)*self.num_features]
            dot_products.append((w_subset*self.train_features).sum(axis=1))

        dot_products = np.array(dot_products).T
        numerators = np.exp(dot_products[np.arange(self.n), self.train_labels])
        denominators = np.sum(np.exp(dot_products), axis=1)
        correct_probs = numerators / denominators
        print(-np.sum(np.log(correct_probs)))

        return -np.sum(np.log(correct_probs))

    def gradient_function(self, w):

        temp_train_features = self.train_features
        temp_phi = self.phi
        temp_n = self.n

        # adjust the data we're working with if mini-batch size is set
        if self.batch_size:
            random_rows = np.random.choice(self.n, self.batch_size, replace=False)
            temp_train_features = temp_train_features[random_rows, :]
            temp_phi = temp_phi[random_rows, :]
            temp_n = self.batch_size


        #first, compute the probabilities
        dot_products = []
        for i in range(self.num_labels):
            w_subset = w[i*self.num_features:(i+1)*self.num_features]
            dot_products.append((w_subset*temp_train_features).sum(axis=1))

        dot_products = np.array(dot_products).T
        numerators = np.exp(dot_products)
        denominators = np.sum(np.exp(dot_products), axis=1)
        probs = np.divide(numerators, denominators.reshape(len(denominators), 1))
        probs_repeated = np.repeat(probs, self.num_features, axis=1)

        expected_counts = np.multiply(probs_repeated, 
                                        np.tile(temp_train_features, self.num_labels))


        return -(temp_phi - expected_counts).sum(axis=0) / temp_n


    def train(self, train_features, train_labels):
        
        self.train_features = train_features
        self.train_labels = train_labels
        self.n = self.train_features.shape[0]
        self.num_features = self.train_features.shape[1]
        self.num_labels = len(set(self.train_labels))
        self.w = np.random.randn(self.num_features*self.num_labels) * self.w_scale

        self.phi = np.zeros((self.n, self.num_features*self.num_labels))
        for i in range(self.n):
            self.phi[i, (self.train_labels[i]*self.num_features):((self.train_labels[i]+1)*self.num_features)] = self.train_features[i, :]

        self.w, final_value, _ = fmin_l_bfgs_b(self.objective_function, self.w, self.gradient_function,
                                                pgtol=self.pgtol)
        print(final_value)


    def predict(self, X):


        dot_products = []
        for i in range(self.num_labels):
            w_subset = self.w[i*self.num_features:(i+1)*self.num_features]
            dot_products.append((w_subset*X).sum(axis=1))

        dot_products = np.array(dot_products).T


        numerators = np.exp(dot_products)
        denominators = np.sum(np.exp(dot_products), axis=1)
        probs = np.divide(numerators, denominators.reshape(len(denominators), 1))

        return np.argmax(probs, axis=1)


if __name__ == "__main__":

    dataset = sys.argv[1]

    output_filename = ""
    dev_output_filename = ""
    output_header = []

    if dataset == 'propernames':
        train_features, train_labels, dev_features, dev_labels, test_features = util.load_propernames()

        label_encodings = util.load_labels_from_array(train_labels)
        label_encodings_reverse = {label_encodings[k]:k for k in label_encodings}

        train_labels = [label_encodings[x] for x in train_labels]
        dev_labels = [label_encodings[x] for x in dev_labels]

        output_filename = "results/maxent_propername_test_predictions.csv"
        dev_output_filename = "data/propernames/dev/dev_pred_maxent.csv"
        output_header = ['id', 'type']

    elif dataset == 'newsgroups':
        train_features, train_labels, dev_features, dev_labels, test_features = util.load_newsgroups()


        label_encodings = util.load_labels_from_array(train_labels)
        label_encodings_reverse = {label_encodings[k]:k for k in label_encodings}

        train_labels = [label_encodings[x] for x in train_labels]
        dev_labels = [label_encodings[x] for x in dev_labels]

        output_filename = "results/maxent_newsgroup_test_predictions.csv"
        dev_output_filename = "data/newsgroups/dev/dev_pred_maxent.csv"
        output_header = ['id', 'newsgroup']

    else:
        print("Argument must be propernames or newsgroups")
        exit()

    print(train_labels)
    model = MaximumEntropyModel()
    model.train(train_features, train_labels)
    predictions = model.predict(train_features)

    predictions_dev = model.predict(dev_features)
    predictions_test = model.predict(test_features)

    print(np.mean(predictions_dev == dev_labels))

    with open(dev_output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        for i in range(predictions_dev.shape[0]):
            writer.writerow([i, label_encodings_reverse[predictions_dev[i]]])

    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        for i in range(predictions_test.shape[0]):
            writer.writerow([i, label_encodings_reverse[predictions_test[i]]])

    