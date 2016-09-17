# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 21:39:50 2016

@author: Jae
"""
'''
In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank, resulting in the deaths
of most of its passengers and crew. Explored a subset of
the RMS Titanic passenger manifest to determine which features best predict whether someone survived 
or did not survive.
Some portion of code is imported from Udacity MLND assignment 0. 
'''
import numpy as np
import pandas as pd

# RMS Titanic data visualization code
from titanic_visualizations import survival_stats

def accuracy_score(truth,pred):
    ''' Return accuracy score for input truth and predictions. '''
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth==pred).mean()*100)
    else:
        return "Number of predictions does not match number of outcomes!"

def predictions_0(data):
    ''' Model with no features. Always predicts a passenger did not survive. '''
    
    predictions = []
    for _, passenger in data.iterrows():
        predictions.append(0)
    # Return out predictions
    return pd.Series(predictions)
    
def predictions_1(data):
    ''' Model with on feature:
            - Predict a passenger survived if they are female. '''
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == "female":
            predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)
    
def predictions_2(data):
    ''' Model with two features:
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are younger than 10. '''
    predictions = []
    for _, passenger in data.iterrows():
        if (passenger["Sex"] == "female") or (passenger["Age"] < 10 and passenger["Sex"] == "male"):
            predictions.append(1)
        else: 
            predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)
  
def predictions_3(data):
    ''' Model with three features:
        - Predict a passenger survived if they are female except for passenger with class > 2 or Sibling/Spouser > 0
        - Predict a passenger survived if they are younger than 10.  '''
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger["Sex"] == "female":
            if passenger["Pclass"] > 2 and passenger["SibSp"] > 0:
                predictions.append(0)
            else: 
                predictions.append(1)
        elif passenger["Age"] < 10:
            predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)
    
    


# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
print full_data.head()

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
print data.head()        

# Test the 'accuracy_score' function. Assume first 5 are survived 
predictions = pd.Series(np.ones(5, dtype = int))  
print accuracy_score(outcomes[:5], predictions)

# Make the predictions - All are NOT survived
predictions = predictions_0(data)
print accuracy_score(outcomes,predictions)

# Visualize the survival outcomes of passengers based on their sex
survival_stats(data,outcomes,'Sex')

# Predict survival rate based on 'Sex')
predictions = predictions_1(data)
print accuracy_score(outcomes,predictions)

# Plot the survival outcomes of male passengers based on their age.
survival_stats(data,outcomes,'Age', ["Sex == 'male'"])

predictions = predictions_2(data)
print accuracy_score(outcomes,predictions)

# Plot the survival outcomes of femae passenger based on passenger class
survival_stats(data,outcomes, 'Pclass', ["Sex == 'female'"])       

predictions = predictions_3(data)
print accuracy_score(outcomes,predictions)

