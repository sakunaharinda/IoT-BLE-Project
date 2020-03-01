

# 1. Importing libraries
import pandas as pd  
import numpy as np
import json
import os
import ast
import csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix 


# 2. Importing the dataset
#Download the dataset from database as a CSV file and store in the local directory. 
#To read data from CSV file, the simplest way is to use read_csv method of the pandas library. 
wifiData = pd.read_csv("wifi_data.csv")


# 3. Exploratory Data Analysis
#check the dimensions of the data and see first few records
print("Dimensions of the data:")
print(wifiData.shape)
print("\nFirst few records:")
print(wifiData.head())

# 4. Data Preprocessing
# To divide the data into attributes and labels
X = wifiData.drop('id', axis=1)  #contains attributes
y = wifiData['id'] # contains coresponding labels

#divide data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.001)  

# 5. Training the Algorithm 
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train) # train the algorithm on the training data

# 6. Making Predictions
#X_test
y_pred = svclassifier.predict(X_test)

# 7. Evaluating the Algorithm
#Confusion matrix, precision, recall, and F1 measures are the most commonly used metrics for classification tasks.
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred)) 

#predict the hidden node

#z= np.array([-50,-89.0,-86.0,-75,-95,-88.0,-75.0,-92.0,-20])
#data_to_predict = z.reshape(1, -1)
#predicted_label = svclassifier.predict(data_to_predict)
#print('Predicted label is %d ' %predicted_label)


unknownData=pd.read_csv("unknownData.csv")
predicted_labels = svclassifier.predict(unknownData)

outputFile = 'PredictedLabels.csv'

for i in range(len(predicted_labels)):
    print("predicted label for data",i,": ",int(predicted_labels[i]))
    
    
    
