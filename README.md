# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages. 
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANDHYA BN
RegisterNumber: 212222040144
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
*/
```

## Output:

To Read Head and Tail Files

![out 2](https://github.com/sandhyabalamurali/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115525118/636f3d3c-0e38-45d8-aa4c-5db428c68021)

Compare Dataset

![out 3](https://github.com/sandhyabalamurali/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115525118/f5fc907c-f184-4dd7-b5e4-a487a240c2a8)

Predicted Value

![out 4](https://github.com/sandhyabalamurali/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115525118/27c0cae3-fc3a-40ae-8b6d-ab668d74dba9)

Graph For Training Set

![out 5](https://github.com/sandhyabalamurali/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115525118/e182c03c-f168-4a5c-b1f7-dc0b5db7bf0d)

Graph For Testing Set

![out 6](https://github.com/sandhyabalamurali/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115525118/a0b2c962-a9a7-4d1f-8b1c-11fbfae780bd)

Error

![out 7](https://github.com/sandhyabalamurali/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115525118/1e1134e2-e5bf-42c4-9dc9-04da16b41b6f)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
