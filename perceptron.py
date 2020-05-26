""" 
Perceptron algorithm
"""
import os
import sys
import pandas as pd
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from util import load_newsgroups, load_propernames
import util

class Perceptron:
    def __init__ (self):
        self.classes = None
        self.classes_dict = None
        self.weights = None
        pass
        
    def fit(self, x_train, y_train, epochs=1):
        self.classes = np.unique(y_train)
        self.classes_dict = {c : i for i,c in enumerate(self.classes)}
        self.weights = np.random.randn(self.classes.shape[0],x_train.shape[1])*0.01
        error = 0
        
        for e in range(epochs):
            for i in range(x_train.shape[0]):   
                #print('row',i)
                arg_max = 0
                predicted_class = 0
                
                for c in range(self.classes.shape[0]):
                    activation = np.dot(self.weights[c], x_train[i])
                    if activation >= arg_max:
                        arg_max = activation
                        predicted_class = c
    
                if (self.classes[predicted_class] != y_train[i]):
                    error += 1
                    self.weights[predicted_class] -= x_train[i]
                    true_class = self.classes_dict.get(y_train[i])
                    self.weights[true_class] += x_train[i]

            
            print('epoch '+ str(e) + '; error ' + str(error))
            error = 0
            
        return self.classes, self.weights
    
    def predict(self, x_test):
        y_pred = []
        for i in range(x_test.shape[0]):   
            #print('row',i)
            arg_max = 0
            predicted_class = 0
            
            for c in range(self.classes.shape[0]-1):
                activation = np.dot(self.weights[c], x_test[i])
                if activation >= arg_max:
                    arg_max = activation
                    predicted_class = c
            
            y_pred.append(self.classes[predicted_class])
        return y_pred

def accuracy(y_dev, y_pred):
    y_dev = np.array(y_dev)
    y_pred = np.array(y_pred)
    return sum(y_pred==y_dev)/len(y_dev)  


#Load data
data_type = sys.argv[1] 

NUM_EPOCHS = 0
if data_type == 'newsgroups':
    run = True
    print("Loading data...")
    train_bow, train_labels, dev_bow, dev_labels, test_bow = load_newsgroups()

    NUM_EPOCHS = 20
    x_train = train_bow
    y_train = train_labels
    x_dev = dev_bow
    y_dev = dev_labels
    x_test = test_bow

    output_filename = "results/perceptron_newsgroup_test_predictions.csv"
    dev_output_filename = "data/newsgroups/dev/dev_pred_perceptron.csv"
    output_header = ['id', 'type']

elif data_type == 'propernames':
    run = True
    print("Loading data...")
    train_bow, train_labels, dev_bow, dev_labels, test_bow = load_propernames()

    label_encodings = util.load_labels_from_array(train_labels)
    label_encodings_reverse = {label_encodings[k]:k for k in label_encodings}

    train_labels = np.array([x for x in train_labels])

    dev_labels = np.array([x for x in dev_labels])

    NUM_EPOCHS = 20
    x_train = train_bow[:, :-3]
    y_train = train_labels
    x_dev = dev_bow[:, :-3]
    y_dev = dev_labels
    x_test = test_bow[:, :-3]

    output_filename = "results/perceptron_propername_test_predictions.csv"
    dev_output_filename = "data/propernames/dev/dev_pred_perceptron.csv"
    output_header = ['id', 'type']
    
else:
    run = False
    print("Add argument 'newsgroups' or 'propernames' to run Perception model on that data set")


if run == True:

    #Train model
    perceptron = Perceptron()
    classes, weights = perceptron.fit(x_train, y_train, epochs=NUM_EPOCHS)

    #Prediction on Dev data
    y_pred = perceptron.predict(x_dev)
    print('Accuracy on Dev data:',accuracy(y_dev, y_pred))
    #Prediction on Test data
    y_pred_test = perceptron.predict(x_test)

    
    with open(dev_output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        for i in range(len(y_pred)):
            writer.writerow([i, y_pred[i]])

    #Writing Test Predictions to CSV
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        for i in range(len(y_pred_test)):
            writer.writerow([i, y_pred_test[i]])

    