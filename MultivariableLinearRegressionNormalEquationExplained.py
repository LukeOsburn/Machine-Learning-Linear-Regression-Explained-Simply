from numpy import *
import numpy as np
import pandas as pd

#Get Data
points = genfromtxt("IQ.csv", delimiter=",",skip_header=1)

#follows on from the previous example
#MultivariableLinearRegressionGradientDescentExplained.py
#Uses same data, calculates the same model equation but by using matrix algebra
#much much faster

#Our model takes the form of: ModelPIQ=theta0+theta1*Brain+theta2*Height+theta3*Weight
#The total error will be the summation of the error for all data points
#The error function for an individual point is the squared difference of the PIQ estimated by our model
#minus the PIQ of the actual data points
#we will now calculate theta0, theta1, theta2, theta3 such that the total error is a minimum
#we will do this by using matrix algebra

#we can represent our model as: [ModelPIQ]=[theta0,theta1,theta2,theta3].[1,Brain,Height,Weight] (Not this is a dotproduct between matrices)
#we have to include a column of 1's into our data matrix in order to include the theta0 value

#The model equation is the same, so is the error function and so is its derivate, it just looks very different when arranged in the form of matrix algerba
#find a derivation of the normal equation here: https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/

#its the same data, with the same model equation so our answers must be the same as from using gradient MultivariableLinearRegressionGradientDescentExplained.py

#utlimately the normal euqation reduces to: theta=(()(X^T).(X))^-1).(X^T)y   Note, '.s' are dot products, ^T are transpose and ^-1 is inverse
#theta is our list of theta values [theta0,theta1,theta2,theta3]
#X is our data matrix [1,Brain,Height,Weight]
#y is our predicted variable, PIQ

#first lets get our data into the right format

df=pd.read_csv("IQ.csv")
PIQ=pd.read_csv("IQ.csv")
#need to create 2 dataframes that are not dependent or linked to each other
#if there is an easier way, please let me know

PIQ=PIQ['PIQ'] #need a single column of "y" values (PIQ)

#we need the first column of data to be only of our xavlues and the first column to be only 1s
X=df
X.loc[:,'PIQ'] = 1
X.columns=['Xo','Brain','Height','Weight'] #lets rename our columns to accurately refelect what we are doing
print("We now have all we need to 'simply' put into the Normal Equation")

#This is all the bits put correctly into the normal equation
#the only input data you need is, X and PIQ as defined
#feel free to break it up to make more sense of it
theta=pd.DataFrame(np.linalg.pinv((X.T).dot(X).values), (X.T).dot(X).columns, (X.T).dot(X).index).dot(X.T).dot(PIQ)
print(theta)
print("These are the correct theta values, same as using gradient descent, if you use enough iterations")
print("excecutes MUCH faster than using gradient descent")
