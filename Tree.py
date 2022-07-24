from IPython.display import display
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("Tumor Cancer Prediction_Data.csv")


display(df.head())

# Cleaning Data From Null and duplicated Values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df.dropna()
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})

display(df.shape)
display(df.head())

x = df.drop(["diagnosis"], axis=1)
y = df["diagnosis"]

scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=100)

tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
tree.fit(x_train, y_train)
prediction = tree.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, prediction))