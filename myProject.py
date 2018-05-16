##### Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('housing.data.txt',delim_whitespace=True,header=None)
#Heading the columns with the named 
col_name=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
dataset.columns=col_name
#Show the data and make a quick review and description
dataset.head()
dataset.describe()
####              Exploring the data 
#importing the library for plotting 
import seaborn as sn
#showing the relation between columes in pairs 
sn.pairplot(dataset,size = 1.5)
plt.show()
#Diminshing the graphs 
col_name_small=['PTRATIO','B','LSTAT','MEDV']
sn.pairplot(dataset[col_name_small],size = 1.5)
plt.show()
#Making a coorelation between variables to check the main variables affecting eachother
pd.options.display.float_format='{:.3f}'.format
data_correlation = dataset.corr()
#visualizing data coorelation
plt.figure(figsize=(14,11))
sn.heatmap(dataset.corr(),annot=True) #for visuallizing a specific columnes we can replace dataset with dataset[col_name_small]'''
plt.show

'''sn.heatmap(dataset[['CRIM','ZN','INDUS','CHAS','MEDV']].corr(),annot = True)
plt.show'''
#                                    Linear regression 
#Making feature and label 
'''X=dataset['RM'].values
Y=dataset['MEDV'].values'''

X=dataset.iloc[:,5].reshape(-1,1)
Y=dataset.iloc[:,13].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#Fitting the regression
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
#regressor.fit(np.transpose(np.matrix(X)),np.transpose(np.matrix(Y)))
regressor.fit(X_train,Y_train)
# Predicting the Test set results
Y_pred = regressor.predict(X_test)
#checking the accuracy 
regressor.score(X_test, Y_test)
regressor.predict(6.5)
#VISUALIZING DATA
plt.figure(figsize = (9,8));
sn.regplot(X,Y);
plt.xlabel('average number of rooms per dwelling')
plt.ylabel('Median value of owner-occupied homes in \$1000')
plt.show();
sn.jointplot(x=dataset['RM'].values,y=dataset['MEDV'].values,data =dataset,kind='reg',size=7)
plt.show()
#fitting another regression between LSTAT and MEDV
X1=dataset['LSTAT'].values
Y1=dataset['MEDV'].values
#splitting data 
from sklearn.cross_validation import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.2, random_state = 0)
#modeling
regressor1=LinearRegression()
regressor1.fit(np.transpose(np.matrix(X1_train)),np.transpose(np.matrix(Y1_train)))#Another way for modeling 
plt.figure(figsize = (12,10));
sn.regplot(X1_train,Y1_train);
plt.xlabel('% lower status of the population')
plt.ylabel('Median value of owner-occupied homes in \$1000')
plt.title('Regression model with train set of data ')
plt.show();
sn.jointplot(x=dataset['LSTAT'].values,y=dataset['MEDV'].values,data =dataset,kind='reg',size=10)
plt.show()


