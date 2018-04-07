# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('housing.data',delim_whitespace=True,header=None)
#Heading the columns with the named 
col_name=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
dataset.columns=col_name
'''dataset.head()'''
dataset.describe()
import seaborn as sn
'''
sn.pairplot(dataset,size = 1.5)
plt.show()
col_name_small=['PTRATIO','B','LSTAT','MEDV']
sn.pairplot(dataset[col_name_small],size = 1.5)
plt.show()'''
#Making a coorelation between variables 
pd.options.display.float_format='{:.3f}'.format
dataset.corr()
plt.figure(figsize=(14,11))
sn.heatmap(dataset.corr(),annot=True)
plt.show
'''sn.heatmap(dataset[['CRIM','ZN','INDUS','CHAS','MEDV']].corr(),annot = True)
plt.show'''
#Linear regression 
#Making feature and label 
'''X=dataset['RM'].values
Y=dataset['MEDV'].values
'''
X=dataset.iloc[:,5].values.reshape(-1,1)
Y=dataset.iloc[:,13].values
#Fitting the regression
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
#regressor.fit(np.transpose(np.matrix(X)),np.transpose(np.matrix(Y)))
regressor.fit(X,Y)

#VISUALIZING DATA
'''plt.figure(figsize = (9,8));
sn.regplot(X,Y);
plt.xlabel('average number of rooms per dwelling')
plt.ylabel('Median value of owner-occupied homes in \$1000')'''
plt.show();
sn.jointplot(x=dataset['RM'].values,y=dataset['MEDV'].values,data =dataset,kind='reg',size=7)
plt.show()
#fitting another regression between LSTAT and MEDV
X1=dataset['LSTAT'].values
Y1=dataset['MEDV'].values
regressor1=LinearRegression()
regressor1.fit(np.transpose(np.matrix(X1)),np.transpose(np.matrix(Y1)))
plt.figure(figsize = (12,10));
sn.regplot(X1,Y1);
plt.xlabel('% lower status of the population')
plt.ylabel('Median value of owner-occupied homes in \$1000')
plt.show();
sn.jointplot(x=dataset['LSTAT'].values,y=dataset['MEDV'].values,data =dataset,kind='reg',size=10)
plt.show()


