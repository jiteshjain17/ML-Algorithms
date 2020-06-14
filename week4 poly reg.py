#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the dataset
dataset = pd.read_csv("C:/Users/jites/Desktop/datasets_88705_204267_Real estate.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#Visualising the data
plt.figure(figsize=(10,8))
plt.subplot(221, title="X2 House age")
sns.distplot(X['X2 house age'])

plt.subplot(222, title="Distance Nearest MRT Station", facecolor = 'y')
sns.distplot(X['X3 distance to the nearest MRT station'])

plt.subplot(223, title="Convenience Stores", facecolor = 'w')
sns.distplot(X['X4 number of convenience stores'])

plt.subplot(224, title="Serial Number", facecolor = 'y')
sns.distplot(X['No'])
plt.show()

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Performance Analysis
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
def reg_score(y_test, y_pred):
    print("RMSE score: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("r2 score: ", r2_score(y_test, y_pred))
    print("MAE score: ", mean_absolute_error(y_test, y_pred))

#Fitting polynomial regression to dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg.score(X_test, y_test)

#predicting test set results
y_pred_poly = lin_reg.predict(X_test)

reg_score(y_test, y_pred_poly)
