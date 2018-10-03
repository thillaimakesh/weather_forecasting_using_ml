# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
def bangalore(targetdata):
    year = float(input('Enter the year to predict the value : '))
    month=int(input('Enter the month to predict: '))
    dfbangalore=pd.read_excel("bangalore.xls",header=0,na_values=['.'],sheetname=month-1)
    dfbangalore.set_index('Year',inplace=True)
    dfbangalore=dfbangalore.loc[1980:2000]
    dfbangalore=dfbangalore[['T','PP','V','F']]
    dfbangalore.interpolate(inplace=True)
    dfbangalore.reset_index(inplace=True)
    features = 'Year'
    target =  targetdata
    x=dfbangalore[features].reshape(-1,1)                 
    y=dfbangalore[target].reshape(-1,1)
  
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=4)
    
    regressor = LinearRegression()
    regressor.fit(x_train,y_train)
  
    y_prediction = regressor.predict(x_test)
    #print("Few Predicted value for the test case : ")
    #print(y_prediction)
  
    slope, intercept = np.polyfit(dfbangalore[features],dfbangalore[target],1)
  
    temp = intercept + (slope*year)
    print('Predicted for the given month and year')
    print(temp)
   
    plt.scatter(x_test,y_test,color='black')
    plt.plot(x_test, slope*x_test + intercept, '-')
    plt.show()


def madurai(targetdata):
    year = float(input('Enter the year to predict the value : '))
    month=int(input('Enter the month to predict: '))
    dfmadurai=pd.read_excel("madurai.xls",sheetname=month-1,header=0,na_values=['-'])
    dfmadurai.set_index('Year',inplace=True)
    dfmadurai=dfmadurai.loc[1980:2000]
  
    dfmadurai=dfmadurai[['T','PP','V','F']]
    dfmadurai.interpolate(inplace=True)
  
    dfmadurai.reset_index(inplace=True)
  
  
    features = 'Year'
    target =  targetdata
    x=dfmadurai[features].reshape(-1,1)                  # To convert to 2D array
    y=dfmadurai[target].reshape(-1,1)
  
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=4)
    
    regressor = LinearRegression()
    regressor.fit(x_train,y_train)
  
    y_prediction = regressor.predict(x_test)
    #print("Few Predicted value for the test case : ")
    #print(y_prediction)
  
    slope, intercept = np.polyfit(dfmadurai[features],dfmadurai[target],1)

    temp = intercept + (slope*year)
    print('The predicted value for the given month and year')
    print(temp)
  
    plt.scatter(x_test,y_test,color='black')
    plt.plot(x_test, slope*x_test + intercept, '-')
    plt.show()


def chennai(targetdata):
    year = float(input('Enter the year to predict the value : '))
    month=int(input('Enter the month to predict: '))
    dfchennai=pd.read_excel("chennai.xls",header=0,na_values=['-'],sheetname=month-1)
    dfchennai.set_index('year',inplace=True)
    dfchennai=dfchennai.loc[1980:1990]
  
    dfchennai=dfchennai[['T','PP','V','F']]
    dfchennai.interpolate(inplace=True)
  
    dfchennai.reset_index(inplace=True)
  
  
    features = 'year'
    target =  targetdata
    x=dfchennai[features].reshape(-1,1)                  # To convert to 2D array
    y=dfchennai[target].reshape(-1,1)
  
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=4)
  
    regressor = LinearRegression()
    regressor.fit(x_train,y_train)
  
    y_prediction = regressor.predict(x_test)
    #print("Few Predicted value for the test case : ")
    #print(y_prediction)
  
    slope, intercept = np.polyfit(dfchennai[features],dfchennai[target],1)
  
    print('The predicted value for the given month and year')
    temp = intercept + (slope*year)
    print(temp)
  
    plt.scatter(x_test,y_test,color='black')
    plt.plot(x_test, slope*x_test + intercept, '-')
    plt.show()


print("\n\n\t\t\t\t\t\tWelcome to Weather Prediction\n\n")

while(True):
    print("1.Predict Bangalore's Weather  2.Predict Madurai's Weather  3.Predict Chennai's Weather ")
    print("Enter the choice to Predict Weather : ")
    choice= int(input())

    if choice==1:
        while(True):
            print("1.Avg Temperature 2. Avg Rainfall 3.Vapour Pressure 4.Fog")
            print("Enter the choice of target data : ")
            targetchoice = int(input())

            if targetchoice==1: 
                bangalore(targetdata='T')
                break
            elif targetchoice==2:
                bangalore('PP')
                break
            elif targetchoice==3:
                bangalore('V')
                break
            elif targetchoice==4:
                bangalore('F')
                break
            else:
                print("Invalid Choice")
                break
        break

    if choice==2:
        while(True):
            print("1.Avg Temperature 2. Avg Rainfall 3.Vapour Pressure 4.Fog")
            print("Enter the choice of target data : ")
            targetchoice = int(input())

            if targetchoice==1:
                madurai('T')
                break
            elif targetchoice==2:
                madurai('PP')
                break
            elif targetchoice==3:
                madurai('V')
                break
            elif targetchoice==4:
                madurai('F')
                break
            else:
                print("Invalid Choice")
                break
        break
    if choice==3:
        while(True):
            print("1.Avg Temperature 2. Avg Rainfall 3.Vapour Pressure 4.Fog")
            print("Enter the choice of target data : ")
            targetchoice = int(input())

            if targetchoice==1:
                chennai('T')
                break
            elif targetchoice==2:
                chennai('PP')
                break
            elif targetchoice==3:
                chennai('V')
                break
            elif targetchoice==4:
                chennai('F')
                break
            else:
                print("Invalid Choice")
                break
        break
    else:
        print("Invalid Choice")