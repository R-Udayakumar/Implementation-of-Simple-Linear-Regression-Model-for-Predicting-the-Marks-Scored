# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:  Udayakumar R
RegisterNumber: 212222230163  
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')

df.head()
df.tail()

x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y

# Splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/2,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

# Displaying predicted values
y_pred
y_test

plt.scatter(x_test,y_test,color = "yellow")
plt.plot(x_train,regressor.predict(x_train),color = "purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title("Hours vs scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()

mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse = np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:
## DATA HEAD 
![image](https://github.com/R-Udayakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708024/2ea7a4a3-d950-4d86-b9cc-076a8f4dae8f)
## DATA TAIL
![image](https://github.com/R-Udayakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708024/d5af93b5-b09f-4d38-952b-9131ae04a683)
## ARRAY VALUES OF X
![image](https://github.com/R-Udayakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708024/4de5f7f4-59e7-43c9-b280-9ee2d521b82f)
## ARRAY VALUES OF Y
![image](https://github.com/R-Udayakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708024/87e43997-f7d0-4b79-8fcd-60073c70a73c)
## VALUES OF Y PREDICTION
![image](https://github.com/R-Udayakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708024/e0741a3d-c635-4099-be77-e8b0a0a87137)
## ARRAY VALUES OF Y TEST
![image](https://github.com/R-Udayakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708024/e55c8f0c-c1ee-4e5d-bc7c-404e87b26863)
## TRAINING SET GRAPH
![image](https://github.com/R-Udayakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708024/8acd153d-c365-46a1-94aa-862572c6c25c)
## TESTING SET GRAPH
![image](https://github.com/R-Udayakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708024/13fc86d1-3126-4d23-b38e-df9848b26536)
## VALUES OF MSE, MAE AND RMSE
![image](https://github.com/R-Udayakumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708024/34d51735-8251-42fc-bf60-74aa0c78902d)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
