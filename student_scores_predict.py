import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('student_scores.csv')
print(dataset.shape)
print('DataSet of Student ScoreSheet')
print(dataset)


X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,:-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.20,random_state=0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(dataset[['course_code']], dataset[['scores']])
user_input = int(input('Enter the CourseCode: '))

print('Prediction value of this courseCode:')
print(model.predict([[user_input]]))

from sklearn.tree import DecisionTreeRegressor

DT_model = DecisionTreeRegressor(max_depth=5).fit(X_train, Y_train)
DT_predict =DT_model.predict(X_test)

print('Some Acurately Predcited Value of this dataset :')
print(DT_predict)

x = dataset['course_code']
y = dataset['scores']
plt.scatter(x, y)

plt.title("Student Score prediction")
plt.xlabel("CourseCode")
plt.ylabel("Percentage Scores")
print('Graph of Students Course VS Scores:')
plt.show()






