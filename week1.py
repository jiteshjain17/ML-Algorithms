import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = pd.read_csv("C:/Users/jites/Desktop/Linear_X_Train week 1")
y_train = pd.read_csv("C:/Users/jites/Desktop/Linear_Y_Train.csv")
X_test = pd.read_csv("C:/Users/jites/Desktop/Linear_X_Test.csv")

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#Visualising the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Student Performance')
plt.xlabel('Time')
plt.ylabel('Score')
plt.show()


