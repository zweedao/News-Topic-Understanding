""" 
Multi-layer Perceptron
"""
import os
import sys
import pandas as pd
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as data_utils
import util


class MLP:
    def __init__ (self, activate_function, epochs, batch_size, learning_rate, momentum, weight_decay):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.model = None
        self.onehot_encoder = None
        self.onehot_columns = None
        self.activate_function = activate_function
        self.loss_history_train = []
        pass
        
    def fit(self, x_train, y_train):
        y_train_onehot = y_train
        self.onehot_columns = y_train_onehot.shape[1]
        
        x_train_tensor = torch.tensor(x_train).float()
        y_train_tensor = torch.tensor(y_train_onehot).float()
        
        train_data_torch = data_utils.TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = data_utils.DataLoader(train_data_torch, batch_size=self.batch_size, shuffle=True)
        
        self.model = nn.Sequential(
            nn.Linear(x_train_tensor.shape[1],200,bias=True),
            #nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            self.activate_function,
            nn.Linear(200,self.onehot_columns,bias=True),
            nn.Softmax(dim=1)    
        )
        print('model',self.model)
        
        loss_function = nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay) 
        
        for i in range(self.epochs):
            loss_total_train = 0
            j = 0
            for input, target in train_loader:
                j +=1
                # Forward Propagation
                y_pred = self.model(input)
                # Compute and print loss
                loss_train = loss_function(y_pred, target)
                loss_total_train += loss_train.item()
                # Zero the gradients
                optimizer.zero_grad()   
                # perform a backward pass (backpropagation)
                loss_train.backward()
                # Update the parameters
                optimizer.step()
                
            lose_avg_train = float(loss_total_train)/j 
            self.loss_history_train.append(lose_avg_train)
            print('epoch:', i,' loss:', lose_avg_train)
            
        return 
    
    def plot_loss(self):
        plt.figure(1)
        plt.plot(range(self.epochs),self.loss_history_train)
        plt.title('Loss over epoch')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    
    def predict(self, x_test):
        x_test_tensor = torch.tensor(x_test).float()        
        prediction_test = self.model(x_test_tensor)

        prediction_test = torch.max(prediction_test,1)[1]
        prediction_test = prediction_test.numpy()
        prediction_test_onehot = np.zeros((prediction_test.size, self.onehot_columns))
        prediction_test_onehot[np.arange(prediction_test.size),prediction_test] = 1
        y_pred_test = np.argmax(prediction_test_onehot, axis=1)
                
        return y_pred_test

def accuracy(y_dev, y_pred):
    y_dev = np.array(y_dev)
    y_pred = np.array(y_pred)
    return sum(y_pred==y_dev)/len(y_dev)  

if __name__ == "__main__":

    dataset = sys.argv[1]

    output_filename = ""
    dev_output_filename = ""
    output_header = []

    if dataset == 'propernames':
        train_features, train_labels, dev_features, dev_labels, test_features = util.load_propernames()

        label_encodings = util.load_labels_from_array(train_labels)
        label_encodings_reverse = {label_encodings[k]:k for k in label_encodings}

        train_labels = np.array([label_encodings[x] for x in train_labels])
        dev_labels = np.array([label_encodings[x] for x in dev_labels])

        output_filename = "results/mlp_propername_test_predictions.csv"
        dev_output_filename = "data/propernames/dev/dev_pred_mlp.csv"
        output_header = ['id', 'type']


    elif dataset == 'newsgroups':
        train_features, train_labels, dev_features, dev_labels, test_features = util.load_newsgroups()

        label_encodings = util.load_labels_from_array(train_labels)
        label_encodings_reverse = {label_encodings[k]:k for k in label_encodings}

        train_labels = np.array([label_encodings[x] for x in train_labels])
        dev_labels = np.array([label_encodings[x] for x in dev_labels])

        output_filename = "results/mlp_newsgroup_test_predictions.csv"
        dev_output_filename = "data/newsgroups/dev/dev_pred_mlp.csv"
        output_header = ['id', 'newsgroup']

    else:
        print("Argument must be propernames or newsgroups")
        exit()


    #converting labels to one-hot representations


    train_labels_one_hot = np.zeros((train_labels.shape[0], len(label_encodings)))
    train_labels_one_hot[np.arange(train_labels.shape[0]), train_labels] = 1
    

    dev_labels_one_hot = np.zeros((dev_labels.shape[0], len(label_encodings)))
    dev_labels_one_hot[np.arange(dev_labels.shape[0]), dev_labels] = 1


    x_train = train_features
    y_train = train_labels_one_hot
    x_dev = dev_features
    y_dev = dev_labels_one_hot
    x_test = test_features

    #Train model

    #activate_func = nn.ReLU()
    #activate_func = nn.Sigmoid()
    activate_func = nn.Tanh() 

    mlp = MLP(activate_function=activate_func, epochs=1000, batch_size=1000000, learning_rate=0.1, momentum=0.9, weight_decay=0)
    mlp.fit(x_train, y_train)
    #mlp.plot_loss()


    #Prediction on Dev data
    y_pred = mlp.predict(x_dev)

    print('Accuracy:',accuracy(dev_labels, y_pred))

    with open(dev_output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        for i in range(y_pred.shape[0]):
            writer.writerow([i, label_encodings_reverse[y_pred[i]]])

    #Writing Test Predictions to CSV
    y_pred_test = mlp.predict(x_test)
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        for i in range(y_pred_test.shape[0]):
            writer.writerow([i, label_encodings_reverse[y_pred_test[i]]])


    