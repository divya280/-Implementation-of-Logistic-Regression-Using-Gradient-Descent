# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression
2. Set variables for assigning dataset values
3. Import linear regression from sklearn.
4. Predict the values of array
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn
6. Obtain the graph

## Program:

/*
### Program to implement the the Logistic Regression Using Gradient Descent.
### Developed by: V.Divyashree
### RegisterNumber:  212223230051
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt", delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:,0],X[y == 1][:,1], label="Admitted")
plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label ="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costfunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return j,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta =np.array([0,0,0])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min()-1, X[:,0].max() +1
  y_min, y_max = X[:,1].min()-1, X[:,1].max() +1
  xx,yy =np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min,y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:,0],X[y== 1][:,1],label="Admitted")
  plt.scatter(X[y== 0][:,0],X[y ==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels =[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
*/


## Output:
Array values of x:

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/e6b2426d-19ad-40e5-969a-9ee81ac53334)

Array values of y

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/75c139ff-4837-4456-a6bc-f4cd13071034)

Exam 1-score graph

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/8b45d334-057f-452c-acb0-487a661e14c1)

Sigmoid function graph

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/531accc8-345a-420a-842c-a1fbcb40c99d)

x_train_grad value

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/109c3367-77bc-4103-a944-f2a514a62d85)

y_train_grad value

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/6e19e331-1d42-4630-afad-f8dc31499157)

Print res.x

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/77c27d55-1358-477c-9327-6da9aca04c10)

Decision boundary - graph for exam score

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/939df1de-cb3f-40e5-817e-bfad885284a8)

Probability value

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/11e955a6-15d3-4eeb-92f9-e7bd568f8fac)

Prediction value of mean

![image](https://github.com/divya280/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/82276099/2b52624e-04f0-4981-a315-fd73a1aa1a07)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

