import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import session
from flask import Flask, render_template
import imp


# Load dataset
dataset = pd.read_csv('data.csv')
# print dataset


new_dataset = dataset.replace(0, np.NaN)
list = ['Unnamed: 32', 'id', 'diagnosis']
new_dataset = new_dataset.drop(list, axis=1)

#print(new_dataset.head(570))

new_dataset = new_dataset.fillna(dataset.mean())
print(new_dataset)

list = ['Unnamed: 32', 'id', 'diagnosis']
X = dataset.drop(list, axis=1)
X.head()

y = dataset.diagnosis

le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

accuracy = knn.score(X_train, y_train)
print(accuracy)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_test, y_test)

y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(accuracy)

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/')
def homepage():
        return render_template('index.html')

@app.route('/home', methods=['POST'])
def predict():
    dataset = pd.read_csv('data.csv')
    testcsv = pd.read_csv('test2.csv')

    x = dataset.iloc[:, 2:-1].values
    y = dataset.iloc[:, 1].values

    testdata = testcsv.iloc[:, :].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(testdata)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    if str(y_pred) == "['B']":
        session['pred'] = str("IT IS BENIGN")
    else:
        session['pred'] = str("IT IS MALIGNANT")


    return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)