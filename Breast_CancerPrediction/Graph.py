import sys

import pandas as pd
import numpy as np
from PyQt4.uic.properties import QtGui
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from PyQt4 import QtCore

class EmittingStream(QtCore.QObject):

    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
def __init__(self, parent=None, **kwargs):
    # ..

    # Install the custom output stream
    sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)

def __del__(self):
    # Restore sys.stdout
    sys.stdout = sys.__stdout__

def normalOutputWritten(self, text):
    """Append text to the QTextEdit."""
    # Maybe QTextEdit.append() works as well, but this is how I do it:
    cursor = self.textEdit.textCursor()
    cursor.movePosition(QtGui.QTextCursor.End)
    cursor.insertText(text)
    self.textEdit.setTextCursor(cursor)
    self.textEdit.ensureCursorVisible()

# This is for plotting decision boundaries taken from textbook.
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('^', 'x', 'o', 's', 'v')
    colors = ('yellow', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
df.columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
df.head()

X = df.iloc[:, 2:31].values
y = df.iloc[:,1].values

df1 = datasets.load_breast_cancer()
x_1= df1.data[:, [1, 3]]
y_1= df1.target
X_train, X_test, y_train, y_test = train_test_split(x_1, y_1, test_size=0.20, random_state=1)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_std, y_train)

scores_knn = cross_val_score(knn, X, y, scoring='recall_macro', cv=5)
print('Accuracy for KNN : %.3f +/- %.3f' % (np.mean(scores_knn), np.std(scores_knn)))
plot_decision_regions(X_combined_std, y_combined, classifier =knn)
plt.title('Decision boundary for KNN ')
plt.xlabel('area_mean & concavity_mean')
plt.ylabel('Diagnosis')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

