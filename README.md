# mielage_prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
mielage = pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/MPG.csv')
mielage.head()
mielage.info()
mielage.describe()
mielage.nunique()
mielage = mielage.dropna()
y = mielage['mpg']
X = mielage[['cylinders','displacement',	'horsepower',	'weight',	'acceleration']]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.intercept_
model.coef_
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
mean_absolute_error(y_test,y_pred)
mean_absolute_percentage_error(y_test,y_pred)
