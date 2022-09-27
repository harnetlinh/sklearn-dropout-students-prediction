import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib import style
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

number_semester = int(sys.argv[2])
input_file = sys.argv[1]
path = "./data/"

# comma delimited is the default
df = pd.read_csv(path + input_file, sep=',', header=None).values


X = []
y = []
# create data set
for row in df:
    _x = []
    for i in range(number_semester * 3 + 1):
        _x.append(row[i])
    X.append(_x)
    y.append(row[number_semester * 3 + 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)   

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Model
svm = SVC(kernel= 'linear', random_state=1, C=0.1)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))