#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("C:/Users/jites/Desktop/Car_Purchasing_Data.csv", encoding='ISO-8859-1')
X = dataset.iloc[:, 2:8].values
y = dataset.iloc[:, 8].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(), [2])], remainder='passthrough')
X = ct.fit_transform(X)

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler(with_mean=False)
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='linear')
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)



