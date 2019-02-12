from numpy import *
import numpy as np
from numpy import *
import pandas as pd

#Get Data
df=pd.read_csv("IQ.csv")

#lets do FEATURE SCALING here
#Lets do the Feature scaling manipuation using Pandas dataframes

#these functions gives the mean and standard deviation of the columns
#print(df['Brain'].mean()) #90.67
#print(df['Brain'].std()) #7.26



#To feature scale, each value in a coulmn= (x-mean)/(std)
#This scales all values to be closer to each other so that gradient descent can work more quickly and reliably

df.Brain = (df.Brain-df['Brain'].mean())/df['Brain'].std()
df.Height = (df.Height-df['Height'].mean())/df['Height'].std()
df.Weight = (df.Weight-df['Weight'].mean())/df['Weight'].std()

points=df.values
#convert dataframe into a numpy array as the original code was written to work on a numpy array input



#We will predict PIQ using a multivariable linear regression model using Brain, Height and Weight
#Our model takes the form of: ModelPIQ=theta0+theta1*Brain+theta2*Height+theta3*Weight

#The total error will be the summation of the error for all data points
#The error function for an individual point is the squared difference of the PIQ estimated by our model
#minus the PIQ of the actual data points

#we will now calculate theta0, theta1, theta2, theta3 such that the total error is a minimum
#we will do this using gradient descent

#initiliazing our theta values and error
tzero=0.0
tone=0.0
ttwo=0.0
tthree=0.0
error=0.0

#we will need to iterate, each time reducing the error based on the partial derivative of the error function
#with respect to each theta value, in turn
iterations=5000

#we multiple the change in error from each iteration by the learning LearningRate
#this is just scaling the change to avoid overshooting
LearningRate=0.001
N = float(len(points))

#Error function:
#For a point, Error=(ModelPIQ-PIQ)^2
#Where ModelPIQ=theta0+theta1*Brain+theta2*Height+theta3*Weight
#we adjust our theta values by taking the average partial derivative of the Error function for each theta value independently
#j=[PIQ,Brain,Height,Weight]
#ModelPIQ/dtheta0=2*(tzero+tone*j[1]+ttwo*j[2]+tthree*j[3]-j[0])
#ModelPIQ/dtheta1=2*(tzero+tone*j[1]+ttwo*j[2]+tthree*j[3]-j[0])*j[1]
#ModelPIQ/dthetan=2*(tzero+tone*j[1]+ttwo*j[2]+tthree*j[3]-j[0])*j[n]
#We can drop the "2*" as its just a scaling

for i in range(iterations):
    sumtzero=0.0
    sumttone=0.0
    sumttwo=0.0
    sumtthree=0.0
    error=0.0
    for j in points:
        #calculate for each data point for each iteration
        sumtzero=sumtzero+(tzero+tone*j[1]+ttwo*j[2]+tthree*j[3]-j[0])
        sumttone=sumttone+(tzero+tone*j[1]+ttwo*j[2]+tthree*j[3]-j[0])*j[1]
        sumttwo=sumttwo+(tzero+tone*j[1]+ttwo*j[2]+tthree*j[3]-j[0])*j[2]
        sumtthree=sumtthree+(tzero+tone*j[1]+ttwo*j[2]+tthree*j[3]-j[0])*j[3]
        #calculate the total error, this should decrease after each iteration
        error=error+(tzero+tone*j[1]+ttwo*j[2]+tthree*j[3]-j[0])**2

    #we adjust the theta values after each iteration
    tzero=tzero-(LearningRate*sumtzero/N)
    tone=tone-(LearningRate*sumttone/N)
    ttwo=ttwo-(LearningRate*sumttwo/N)
    tthree=tthree-(LearningRate*sumtthree/N)
    #print(error)




#these are our final theta values and final error
#print("ERROR")
#print(error)
#print("TZERO")
#print(tzero)
#print("TONE")
#print(tone)
#print("TTWO")
#print(ttwo)
#print("TTHREE")
#print(tthree)

#rounding
tzero=round(tzero, 3)
tone=round(tone, 3)
ttwo=(round(ttwo, 3))
tthree=(round(tthree, 3))

print("rescaled brain")
tone=(round(tone, 3)*df['Brain'].std())+df['Brain'].mean()
print(tone)

print("Our final model is: PIQ=%s + %s*Brain %s*Height%s*Weight" % (tzero, tone,ttwo,tthree))
print("Increase the number of iterations to get a better model")
print("Note the theta values are different comapared to when we didn't use feature scaling")
print("Need to use the scaled input values to get the same predictions")
firstPIQ=round(tzero+(points[0,1]*tone)+(points[0,2]*ttwo)+(points[0,3]*tthree),3)
print("The predicted PIQ value for the first data point is %s" %(firstPIQ))
print("Note, this is the same or close enough to be considered the same from the other methods")
print("As the iterations increase, the results will tend closer to each other")
