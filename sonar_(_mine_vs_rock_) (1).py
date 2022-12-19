# -*- coding: utf-8 -*-
"""Sonar ( Mine VS Rock ).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UGFuSaq4_1ws4A-PSSQmn467KBGaL6D6

IMPORTING THE MODULES
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""DATA COLLECTION AND DATA PROCESSING"""

#Loading Dataset to the Panda Dataframes
rock_data = pd.read_csv("/content/ROCK_OR_MINE.csv",header=None)

rock_data.head(5)

# .shape TELLS THE ROWS AND COLUMN OF THE DATASET.
rock_data.shape

#.describe TELLS THE STATICAL DATA.
rock_data.describe()

#.value_counts TELLS THE ROCK AND MINE DATA
rock_data[60].value_counts()

"""M : Mine 
R : Rock
"""

rock_data.groupby(60).mean()

"""Gives the mean value for the each and every row for both Rock and Mine."""

#Seperating the data and the labels.
#Supervised learning.
#Training machine learning model.
#Dropping column then axis is 1 and for row axis is 0.

x=rock_data.drop(columns=60 , axis=1)

y=rock_data[60]

print(x)
print(y)

"""Training and Testing data

1. The train-test split : Is a technique for evaluating the performance of a machine learning algorithm.

Taking a dataset and dividing it into two subsets:

1. Train Dataset: Used to fit the machine learning model.
2. Test Dataset: Used to evaluate the fit machine learning model.

test_size : Means the percentage of data needed for testing the data. ( here 10% of data is taken to test the data. )

stratify : Means to split the data in the accurate ratio of the data given. ( here it means the ratio of the test data of the rock is to mine should be almost same. )

random_state : Means to split the data in the [articular order.
"""

#from pandas.core.common import random_state
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.1,stratify=y ,random_state = 1)

print(x.shape,x_train.shape,x_test.shape)

print(x_train)
print(y_train)

"""Training a model : Using Logistic Regression"""

model = LogisticRegression()

#tarining the Logistic Regression model with training data.
model.fit(x_train , y_train)

"""Model Evaluation"""

#Accuracy on the training data.
#x_train_prediction : Is our prediction of the accuracy of the data.
#y_train : Is the real accuracy of the data.
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print("Accuracy on the training data : " , training_data_accuracy)

"""83% is the accuracy of the training data."""

#Accuracy on the testing data.
#x_test_prediction : Is our prediction of the accuracy of the data.
#y_test : Is the real accuracy of the data.
x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("Accuracy on the testing data : " , testing_data_accuracy)

"""76% is the accuracy of the testing data.
( The accuracy means that out of 100 times 76 times it will tell whether it is a rock or mine correctly.)

Making Predictive System

( Predict whether the object is Rock or Mine using the data.)
"""

#input_data = ()
input_data = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)

#Changing the input_data to a numpy array.

input_data_as_numpy_array = np.asarray(input_data)

#Reshaping the Numpy array as we are predicting for one instance.

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#(1,-1) represent the 1 instance and predicting the label for that one instance.

prediction = model.predict(input_data_reshaped)

#Prediction will return whether the object is R i.e. Rock or M i.e. Mine.

print(prediction)

if ( prediction[0] == 'R') :
  print("The object is a Rock.")

else :
  print("The object is a Mine.")

#input_data = ()
input_data = (0.0654,0.0649,0.0737,0.1132,0.2482,0.1257,0.1797,0.0989,0.2460,0.3422,0.2128,0.1377,0.4032,0.5684,0.2398,0.4331,0.5954,0.5772,0.8176,0.8835,0.5248,0.6373,0.8375,0.6699,0.7756,0.8750,0.8300,0.6896,0.3372,0.6405,0.7138,0.8202,0.6657,0.5254,0.2960,0.0704,0.0970,0.3941,0.6028,0.3521,0.3924,0.4808,0.4602,0.4164,0.5438,0.5649,0.3195,0.2484,0.1299,0.0825,0.0243,0.0210,0.0361,0.0239,0.0447,0.0394,0.0355,0.0440,0.0243,0.0098)

#Changing the input_data to a numpy array.

input_data_as_numpy_array = np.asarray(input_data)

#Reshaping the Numpy array as we are predicting for one instance.

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#(1,-1) represent the 1 instance and predicting the label for that one instance.

prediction = model.predict(input_data_reshaped)

#Prediction will return whether the object is R i.e. Rock or M i.e. Mine.

print(prediction)

if ( prediction[0] == 'R') :
  print("The object is a Rock.")

else :
  print("The object is a Mine.")