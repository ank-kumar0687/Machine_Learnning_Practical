
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df_wine=pd.read_csv('wine.csv')
print(df_wine)

print('Cultivars', np.unique(df_wine['Cultivars']))

print(df_wine.head())

x=df_wine.iloc[:,1:].values
print(x)

y=df_wine.iloc[:,0].values
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


