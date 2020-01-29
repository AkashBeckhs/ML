#%%
import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn import model_selection
import sklearn.feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


df = pd.read_csv("C:\\Users\\aakash\\Documents\\GIT_WS\\ML\\data-processing\\adult_income.csv")



#look at the outcome variable #salary
print(df['salary'].value_counts())

#assign outcome as 0 if salary is <=50k and 1 if salary is >50k
df['salary']= [0 if  '<=50K' in x else 1 for x in df['salary']]


#Assign x as dataframe of features and y as corresponding outputs salary
x=df.drop('salary',1)
y=df.salary

#basic data cleaning --
"""
  Delaing with data types:
  1> Numeric        : income,age
  2>Oridnal         : Size X<XL<XXL  
  3>Categorical     : sex,nationality

  Models can only handle numerical values

  Must convert categorical and ordinal features into numeric values
  1> Create dummy features
  2>Transform categorical features into a set of dummy features , each representing unique values
  3>In the set of dummy features 1 indicates the obervation belongs to that category

  Ex : make three fetures(colums) X XL and XXL and 1,0,0 signifies obervation is X

  
"""

#aggregating native-country in two parts united-states and others

#print(x['native_country'].value_counts())
x['native_country'] =['United-States' if x=='United-States' else 'Others' for x in x['native_country']]
#print(x['native_country'].value_counts())

x.replace('NaN',np.NaN)



toDummyList=['workclass','education','marital_status','occupation','relationship','race','sex','native_country']

def dummyDf(df,toDummyList):
  for x in toDummyList:
    print('for '+x)
    dummies=pd.get_dummies(df[x],prefix=x,dummy_na=False)
    df=df.drop(x,1)
    df=pd.concat([df,dummies],axis=1)
  return df


x=dummyDf(x,toDummyList)

#print(x.head(5))

#---Handling missing values

#print(x.isna().sum().sort_values(ascending=False).head())

imp =SimpleImputer(strategy='mean',missing_values=np.NaN)
imp.fit(x)
x=pd.DataFrame(data=imp.transform(x),columns=x.columns)


#print(x.isna().sum().sort_values(ascending=False).head())
#---Detecting outliers using Turkey IQR

"""
Turkey IQR
  Detects Outliers
  Outliers are defined as Q1-1.5(Q3-Q1) or above Q3+1.5(Q3-Q1) Where Q1 and Q3 are quartile 1 and 3

Standart deviation from mean is another approach to detect extreme values.

"""

def findOutliersUsingTurkey(x):
  q1=np.percentile(x,25)
  q3=np.percentile(x,75)
  iqr=q3-q1
  floor=q1-1.5*(iqr)
  ceiling=q3+1.5*(iqr)
  outlierIndices=list(x.index[(x<floor)|(x>ceiling)])
  outlierValues=list(x[outlierIndices])
  return outlierIndices,outlierValues


turkeyIndices,turkeyValues=findOutliersUsingTurkey(x['age'])
#print(np.sort(turkeyValues))


#Kernel Desnsity estimation can also be used to find outliers, Here I am only focusing on steps not techniques

def plotHistogram(x):
  plt.hist(x,color='grey',alpha=0.5)
  plt.title("Histogram of '{var_name}'".format(var_name=x.name)) 
  plt.xlabel("Value")
  plt.ylabel("Frequency")
  plt.show()

#plotting histogram over dependent variable
def plotHistogramDV(x,y):
  plt.hist(list(x[y==0]),alpha=0.5,label='DV=0')
  plt.hist(list(x[y==1]),alpha=0.5,label='DV=1')
  plt.title("Histogram of '{var_name}'".format(var_name=x.name))
  plt.xlabel("Value")
  plt.ylabel("Frequency")
  plt.legend(loc='upper right')  
  plt.show()


plotHistogram(x['age'])
plotHistogramDV(x['age'],y)


#---Feature Engineering (Dimensionality reduction and addition)

"""
Adding new features to exiting data set can cause any problems like increase in data size and over fitting of model, better to use domain knowledge. 
Use interaction b/w two variables when required(requires domain knowledge)
"""

def add_interactions(df):
  combos=list(combinations(list(df.columns),2))
  colNames=list(df.columns) + ['_'.join(x) for x in combos]

  #Find interactions
  poly =PolynomialFeatures(interaction_only=True,include_bias=False)
  df=poly.fit_transform(df)
  df=pd.DataFrame(df)
  df.columns= colNames

  nointIndices=[i for i ,x in enumerate(list((df==0).all())) if x]
  df=df.drop(df.columns[nointIndices],axis=1)

  return df

x=add_interactions(x)
#print(x.head(5))

"""
Dimensionality Reduction

Reduces the number of columns. Very useful when there are lots of features and co-related variables

PCA is one common technique 
  Each principal component is calculated by finding the linear combinationof feature that maximizes variance while also ensuring zero corealtion with the previously calculated principal components
  Downfall is it is hard to interpret.


"""

pca=PCA(n_components=10)
XPca =pd.DataFrame(pca.fit_transform(x))

print(XPca.head(5))


#--- Building model

"""
 Splitting data into training and validation sets

"""

xTrain,Xtest,yTrain,yTest=model_selection.train_test_split(x,y,train_size=0.70,random_state=1)

#selecting features 
select =sklearn.feature_selection.SelectKBest(k=50)
selectFeatures =select.fit(xTrain,yTrain)
indicesSelected=selectFeatures.get_support(indices=True)
columnNamesSelected = [x.columns[i] for i in indicesSelected]

xTrainSelected=xTrain[columnNamesSelected]
xTestSelected=Xtest[columnNamesSelected]

print(columnNamesSelected)

def findModelPref(xTrain,xTest,yTrain,yTest):
  model=LogisticRegression()
  model.fit(xTrain,yTrain)
  yHat=[x[1] for x in model.predict_proba(xTest)]
  auc=roc_auc_score(yTest,yHat)
  return auc

aucProcessed = findModelPref(xTrainSelected,xTestSelected,yTrain,yTest)
print(aucProcessed)