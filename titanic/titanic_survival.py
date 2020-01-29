#%%
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.decomposition import PCA
from sklearn import model_selection
import sklearn.feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


df = pd.read_csv("C:\\Users\\aakash\\Documents\\GIT_WS\ML\\titanic\\train.csv")

df=df.drop("PassengerId",1).drop("Name",1).drop("Ticket",1).drop('Cabin',1)
x=df.drop("Survived",1)
y=df.Survived


dummyList=['Embarked','Sex']

def dummyDf(df,toDummyList):
  for x in toDummyList:
    dummies=pd.get_dummies(df[x],prefix=x,dummy_na=False)
    df=df.drop(x,1)
    df=pd.concat([df,dummies],axis=1)
  return df

x=dummyDf(x,dummyList)

imp =SimpleImputer(strategy='mean',missing_values=np.NaN)
imp.fit(x)
x=pd.DataFrame(data=imp.transform(x),columns=x.columns)


print(x.isna().sum().sort_values(ascending=False))

xTrain,Xtest,yTrain,yTest=model_selection.train_test_split(x,y,train_size=0.70,random_state=1)

def findModelPref(xTrain,xTest,yTrain,yTest):
  model=LogisticRegression()
  model.fit(xTrain,yTrain)
  yHat=[x[1] for x in model.predict_proba(xTest)]
  auc=roc_auc_score(yTest,yHat)
  return auc


aucProcessed = findModelPref(xTrain,Xtest,yTrain,yTest)
print("After adding interactions %d" %aucProcessed)





