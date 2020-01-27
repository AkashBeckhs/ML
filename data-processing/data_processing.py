#%%
import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling



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



toDummyList=['age','workclass','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country']

def dummyDf(df,toDummyList):
  for x in toDummyList:
    print('for '+x)
    dummies=pd.get_dummies(df[x],prefix=x,dummy_na=False)
    df=df.drop(x,1)
    df=pd.concat([df,dummies],axis=1)
  return df


x=dummyDf(x,toDummyList)

print(x.head(5))

#Handling missing values

print(x.isna().sum().sort_values(ascending=False).head())

imp =SimpleImputer(strategy='mean',missing_values=np.NaN)
imp.fit(x)
x=pd.DataFrame(data=imp.transform(x),columns=x.columns)


print(x.isna().sum().sort_values(ascending=False).head())
# Detecting outliers using Turkey IQR

"""
Turkey IQR
  Detects Outliers
  Outliers are defined as Q1-1.5(Q3-Q1) or above Q3+1.5(Q3-Q1) Where Q1 and Q3 are quartile 1 and 3

Standart deviation from mean is another approach to detect extreme values.

"""

def findOutliersUsingTurkey(x):
  q1=np.percentile(x,25)
  q2=np.percentile(x,75)
  iqr=q3-q1
  floor=q1-1.5*(iqr)
  ceiling=q3+1.5*(iqr)
  outlierIndices=list(x.index[(x<floor)|(x>ceiling)])
  outlierValues=list(x[outlierIndices])
  return outlierIndices,outlierValues


turkeyIndices,turkeyValues=findOutliersUsingTurkey(x['age'])
print(np.sort(turkeyValues))







