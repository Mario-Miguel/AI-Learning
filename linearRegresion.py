import pandas as pd;
import numpy as np;
import sklearn;
from sklearn import linear_model;
from sklearn.utils import shuffle;
 

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

#This makes a selection of the fields we want to now
data =data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())


#Label-> attribute we want to predict
predict = "G3"

#One array to store all features
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


#Set data used to training and data used to testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Model to predict student's grades
linear = linear_model.LinearRegression()

#Train model with data 
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

print(accuracy)

#Print constants to generate line (y = m*x+b)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

#Show predictions for each student
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

