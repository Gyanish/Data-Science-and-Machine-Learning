#Importing the required modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Reading "letterCG.bin" file through Pandas's read method
#The file is a binary file which contains attributes of letters in columns 

letter=pd.read_csv("letterCG.bin",sep=' ')

#Received unnecessary column in the data frame "Unnamed: 18" hence removing that column from the letter data frame

letter.head()
letter.drop('Unnamed: 18',inplace=True,axis=1)
letter.head()
letter.columns

#Setting the columns names in correct order to the letter data frame

columns=['Class', 'x-box', 'y-box', 'width', 'high', 'onpix','x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege',
       'xegvy', 'y-ege', 'yegvx', 'x']

letter.columns=columns
letter.head()
letter.drop('x',inplace=True,axis=1)
letter.head()
letter.Class.unique()

#Converting the categorical variable in 1's and 0's

letter.Class=letter.Class.map({'C':1,'G':0})
letter.head()

#Dividing the data set into x and y which contains all the independent variable and dependent variable respectively

x=letter.iloc[:,1:]
y=letter.iloc[:,0]

#Splitting the data set into train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Importing and fitting Decision Tree algorithm on the training set

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(max_depth=2,random_state=0)

dtc.fit(x_train,y_train)

#Predicting the output with Decision Tree Classifier

y_pred=dtc.predict(x_test)

#Checking the accuracy

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

#Result= 0.8774834437086093

#Applying Adaptive boosting to train the week learners in the training data set to increase the predicting capability.

from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier(base_estimator=dtc,n_estimators=16)
abc.fit(x_train,y_train)
y_pred1=abc.predict(x_test)
accuracy_score(y_test,y_pred1)

#Result=0.9867549668874173
