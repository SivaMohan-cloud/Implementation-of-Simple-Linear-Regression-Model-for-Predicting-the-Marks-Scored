# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SIVAMOHANASUNDARAM.V
RegisterNumber:  212222230145

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/Untitled spreadsheet - Sheet1.csv')
df.head()
df.tail()
#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#displaying predicted values
y_pred
#displaying actual values
y_test
#graph plot for training data
plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
df.head(): ![image](https://github.com/SivaMohan-cloud/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418870/3874b5e7-9f3a-4138-b680-23a41ebb4c3b)


df.tail(): ![image](https://github.com/SivaMohan-cloud/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418870/8caddbee-a0e9-4a7f-8799-1722bec6a447)


Array value of X: ![image](https://github.com/SivaMohan-cloud/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418870/9a500e05-469d-4bf2-b97f-7d32f98f783d)


Array value of Y: ![image](https://github.com/SivaMohan-cloud/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418870/2873eeaf-fb56-4345-b3cd-206f7e482d34)


Values of Y prediction: ![image](https://github.com/SivaMohan-cloud/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418870/fe8ccdc3-29a4-4a97-9970-59a427e1555a)


Values of Y test: ![image](https://github.com/SivaMohan-cloud/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418870/7a5c7f91-0108-47bc-9b53-8e0d0404b25f)


Training Set Graph: ![image](https://github.com/SivaMohan-cloud/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418870/ef90df5c-e119-482f-8dd5-b880a358727e)


Test Set Graph: ![image](https://github.com/SivaMohan-cloud/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418870/935d1f2a-1649-498c-8e6b-da3d5164cfad)


Values of MSE, MAE and RMSE: ![image](https://github.com/SivaMohan-cloud/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121418870/857d1733-b57b-4dc3-82ae-c162553a7f5b)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
