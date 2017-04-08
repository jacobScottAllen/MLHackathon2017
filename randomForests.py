from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Data preprocessing
# ------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Save the Id's from the test set (we need them for the predictions)
id_test = test['Id']
X_test = test.drop('Id', axis=1)

y_train = train['Alien_Type']
X_train = train.drop(['Id', 'Alien_Type'], axis=1)

# New split for training and evaluation from training set.
X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.3)

# Model fitting and Predictions
# ------------------
clf = RandomForestClassifier(n_estimators=100, random_state=1)

clf.fit(X_train, y_train)

val_pred = clf.predict(X_val)

acc = accuracy_score(y_val, val_pred)
print ("Multiclass accuracy: {}".format(acc))

X_train = pd.concat([X_train, X_val])
y_train = pd.concat([y_train, y_val])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Creating a submission
# ------------------
submission = pd.DataFrame({"Id":id_test, "Alien_Type":y_pred})
submission.to_csv("submission.csv", index=False)
