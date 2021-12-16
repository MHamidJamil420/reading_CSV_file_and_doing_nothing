from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import csv
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# X, y = load_iris(return_X_y=True)
data = pd.read_csv('sample16_1.csv')

gend = {'M': 1, 'F': 0}
data.Sex = [gend[gender] for gender in data.Sex]

bp = {"HIGH": 1, "LOW": 0, "NORMAL": 2}
data.BP = [bp[temp] for temp in data.BP]

ch = {"HIGH": 1, "LOW": 0, "NORMAL": 2}
data.Cholesterol = [ch[temp] for temp in data.Cholesterol]

drg = {"DrugY": 0, "drugX": 1, "drugC": 2, "drugA": 3, "drugB": 4}
data.Drug = [drg[temp] for temp in data.Drug]

print(data)

X = data.drop('Drug', axis=1)
Y = data.Drug
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

results = []
gnb = GaussianNB()
nc = NearestCentroid()
rt = tree.DecisionTreeRegressor()

rtPrediction = rt.fit(X_train, y_train).predict(X_test)

ncPrediction = nc.fit(X_train, y_train).predict(X_test)
# for training and and testing of a function
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != y_pred).sum()))
results.append(((y_test != y_pred).sum() / X_test.shape[0]) * 100)

print("Number of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != ncPrediction).sum()))
results.append(((y_test != ncPrediction).sum() / X_test.shape[0]) * 100)

print("Number of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != rtPrediction).sum()))
results.append(((y_test != rtPrediction).sum() / X_test.shape[0]) * 100)

print(X.shape, len(Y))
print(X_train.shape, y_train.shape, X_test.shape)

plot_confusion_matrix(gnb, X_test, y_test)
plt.show()
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(['GaussianNB', 'NearestCentroid', 'regression'], results)
