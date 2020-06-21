#Importing Libraries
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

#Importing datasets
X_train = pd.read_csv("C:/Users/jites/Desktop/Diabetes_XTrain.csv")
y_train = pd.read_csv("C:/Users/jites/Desktop/Diabetes_YTrain.csv")
X_test = pd.read_csv("C:/Users/jites/Desktop/Diabetes_Xtest.csv")

#Checking data head and info
print(X_train.info(), X_train.head())
print(X_train.isnull())

#Data Visualizing for each independent variable
#Pregnancies
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.hist(X_train['Pregnancies'][y_train['Outcome'] == 0], bins=15, alpha=0.7, label='Y = 0')
plt.hist(X_train['Pregnancies'][y_train['Outcome'] == 1], bins=15, alpha=0.7, label='Y = 1')
plt.ylabel('Distribution')
plt.xlabel('Pregnancies')
plt.legend()
plt.show()

#Glucose
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 2)
plt.hist(X_train['Glucose'][y_train['Outcome'] == 0], bins=15, alpha=0.7, label='Y = 0')
plt.hist(X_train['Glucose'][y_train['Outcome'] == 1], bins=15, alpha=0.7, label='Y = 1')
plt.ylabel('Distribution')
plt.xlabel('Glucose')
plt.legend()
plt.show()

#Blood Pressure
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.hist(X_train['BloodPressure'][y_train['Outcome'] == 0], bins=15, alpha=0.7, label='Y = 0')
plt.hist(X_train['BloodPressure'][y_train['Outcome'] == 1], bins=15, alpha=0.7, label='Y = 1')
plt.ylabel('Distribution')
plt.xlabel('Blood Pressure')
plt.legend()
plt.show()

#Skin Thickness
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 2)
plt.hist(X_train['SkinThickness'][y_train['Outcome'] == 0], bins=15, alpha=0.7, label='Y = 0')
plt.hist(X_train['SkinThickness'][y_train['Outcome'] == 1], bins=15, alpha=0.7, label='Y = 1')
plt.ylabel('Distribution')
plt.xlabel('SkinThickness')
plt.legend()
plt.show()

#Insulin
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.hist(X_train['Insulin'][y_train['Outcome'] == 0], bins=15, alpha=0.7, label='Y = 0')
plt.hist(X_train['Insulin'][y_train['Outcome'] == 1], bins=15, alpha=0.7, label='Y = 1')
plt.ylabel('Distribution')
plt.xlabel('Insulin')
plt.legend()
plt.show()

#BMI
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 2)
plt.hist(X_train['BMI'][y_train['Outcome'] == 0], bins=15, alpha=0.7, label='Y = 0')
plt.hist(X_train['BMI'][y_train['Outcome'] == 1], bins=15, alpha=0.7, label='Y = 1')
plt.ylabel('Distribution')
plt.xlabel('BMI')
plt.legend()
plt.show()

#DiabetesPedigreeFunction
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.hist(X_train['DiabetesPedigreeFunction'][y_train['Outcome'] == 0], bins=15, alpha=0.7, label='Y = 0')
plt.hist(X_train['DiabetesPedigreeFunction'][y_train['Outcome'] == 1], bins=15, alpha=0.7, label='Y = 1')
plt.ylabel('Distribution')
plt.xlabel('Diabetes Pedigree Function')
plt.legend()
plt.show()

#Age
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 2)
plt.hist(X_train['Age'][y_train['Outcome'] == 0], bins=15, alpha=0.7, label='Y = 0')
plt.hist(X_train['Age'][y_train['Outcome'] == 1], bins=15, alpha=0.7, label='Y = 1')
plt.ylabel('Distribution')
plt.xlabel('Age')
plt.legend()
plt.show()

#Replacing 0 values by mean values
X_train = X_train.replace(0, np.mean(X_train))
X_test = X_test.replace(0, np.mean(X_test))

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Fitting KNN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(metric='minkowski', p=2, n_neighbors=5)
classifier.fit(X_train, y_train.values.ravel())

#Checking score on training set
print(classifier.score(X_train, y_train))

#Checking results on training set
y_pred = classifier.predict(y_train)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
print(cm)





