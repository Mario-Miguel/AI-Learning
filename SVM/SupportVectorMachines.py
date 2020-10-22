import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# Load data from sklearn
cancer = datasets.load_breast_cancer()

# print("Features: ", cancer.feature_names)
# print("Labels: ", cancer.target_names)


x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# print(x_train[:5], y_train[:5])

# C is the margin, number of points allowed to be where they shouldn't in the plane
# Kernel is the function used to change from N dimension of data to N+1
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc =metrics.accuracy_score(y_test, y_pred)
print(acc)