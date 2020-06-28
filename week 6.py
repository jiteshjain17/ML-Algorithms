#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing dataset
dataset = pd.read_csv("C:/Users/jites/Desktop/pulsar_stars.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Info about dataset
print("data info: ", dataset.info())

#Missing values
print(dataset.isnull().sum())

#Statistical info about dataset
print(dataset.describe())

#Visual EDA
#Correlation between variables
correlation = dataset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap=sns.color_palette("magma"), linewidth=2, edgecolor='k')
plt.title("Correlation between variables")
plt.show()

#Pairplot
sns.pairplot(dataset, hue='target_class', palette='husl', diag_kind='kde', kind='scatter')
plt.show()

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X. fit_transform(X_test)

#Fitting logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("lr score: ", lr.score(X_test, y_test))

#Fitting KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("knn score: ", knn.score(X_test, y_test))

#Fitting SVM
from sklearn.svm import SVC
svm = SVC(kernel='poly', degree=3, random_state=1)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM score: ", svm.score(X_test, y_test))

#Fitting Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy', random_state=1)
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
print("Decision Tree score: ", dtc.score(X_test, y_test))

#Fitting Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=1)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
print("Random Forest score: ", rfc.score(X_test, y_test))

#Fitting Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Naive Bayes score: ", nb.score(X_test, y_test))

#Confusion matrix for different models
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_dtc = confusion_matrix(y_test, y_pred_dtc)
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
cm_nb = confusion_matrix(y_test, y_pred_nb)

#Evaluating Model by confusion matrix
plt.figure(figsize=(24,12))
plt.suptitle("Confusion Matrices", fontsize=24)

plt.subplot(2,3,1)
plt.title('Logistic Regression Confusion Matrix')
sns.heatmap(cm_lr, cbar=False, annot=True, cmap='CMRmap_r', fmt='d')

plt.subplot(2,3,2)
plt.title('KNN Confusion Matrix')
sns.heatmap(cm_knn, cbar=False, annot=True, cmap='CMRmap_r', fmt='d' )

plt.subplot(2,3,3)
plt.title('SVM Confusion Matrix')
sns.heatmap(cm_svm, cbar=False, annot=True, cmap='CMRmap_r', fmt='d' )

plt.subplot(2,3,4)
plt.title('Decision Tree Confusion Matrix')
sns.heatmap(cm_dtc, cbar=False, annot=True, cmap='CMRmap_r', fmt='d' )

plt.subplot(2,3,5)
plt.title('Random Forest Confusion Matrix')
sns.heatmap(cm_rfc, cbar=False, annot=True, cmap='CMRmap_r', fmt='d' )

plt.subplot(2,3,6)
plt.title('Naive Bayes Confusion Matrix')
sns.heatmap(cm_nb, cbar=False, annot=True, cmap='CMRmap_r', fmt='d' )

plt.show()


