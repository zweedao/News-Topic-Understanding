""" 
Multi-layer Perceptron for Proper Name data set
"""
import os
import sys
import pandas as pd
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as data_utils
from util import load_newsgroups, load_propernames


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
        self.onehot_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
        self.onehot_encoder.fit(y_train.values)
        y_train_onehot = self.onehot_encoder.transform(y_train.values)
        self.onehot_columns = y_train_onehot.shape[1]
        
        x_train_tensor = torch.tensor(x_train.values).float()
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
        x_test_tensor = torch.tensor(x_test.values).float()        
        prediction_test = self.model(x_test_tensor)

        prediction_test = torch.max(prediction_test,1)[1]
        prediction_test = prediction_test.numpy()
        prediction_test_onehot = np.zeros((prediction_test.size, self.onehot_columns))
        prediction_test_onehot[np.arange(prediction_test.size),prediction_test] = 1
        y_pred_test = self.onehot_encoder.inverse_transform(prediction_test_onehot)
                
        return y_pred_test

def accuracy(y_dev, y_pred):
    y_dev = np.array(y_dev)
    y_pred = np.array(y_pred)
    return sum(y_pred==y_dev)/len(y_dev)  



#Load data
print("Loading data...")
train_bow, train_labels, dev_bow, dev_labels, test_bow = load_propernames()

x_train = pd.DataFrame(train_bow)
y_train = pd.DataFrame(train_labels)
x_dev = pd.DataFrame(dev_bow)
y_dev = pd.DataFrame(dev_labels)
x_test = pd.DataFrame(test_bow)


#Train model

#activate_func = nn.ReLU()
#activate_func = nn.Sigmoid()
activate_func = nn.Tanh() 

mlp = MLP(activate_function=activate_func, epochs=5000, batch_size=1000000, learning_rate=0.1, momentum=0.9, weight_decay=0)
mlp.fit(x_train, y_train)
#mlp.plot_loss()


#Prediction on Dev data
y_pred = mlp.predict(x_dev)
print('Accuracy:',accuracy(y_dev, y_pred))

y_pred = pd.DataFrame(y_pred).reset_index()
pd.DataFrame(y_pred).to_csv(index=False, header=['id','type'], path_or_buf="./data/propernames/dev/dev_pred_mlp.csv")


#Prediction on Test data
y_pred_test = mlp.predict(x_test)

y_pred_test = pd.DataFrame(y_pred_test).reset_index()
pd.DataFrame(y_pred_test).to_csv(index=False, header=['id','type'], path_or_buf="./results/mlp_propername_test_predictions.csv")


