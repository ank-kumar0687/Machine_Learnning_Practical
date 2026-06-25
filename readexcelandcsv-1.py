import pandas as pd

winedatac=pd.read_csv('wine.csv')

print(winedatac)
print(winedatac.head())
print("shape\n",winedatac.shape)
print("columns\n",winedatac.columns)
print("dtypes\n",winedatac.dtypes)
print("ndim\n",winedatac.ndim)
print("size\n",winedatac.size)


winedatae=pd.read_excel('wine.xls')

print('\n')

print(winedatae)
print(winedatae.head())
print("shape\n",winedatae.shape)
print("columns\n",winedatae.columns)
print("dtypes\n",winedatae.dtypes)
print("ndim\n",winedatae.ndim)
print("size\n",winedatae.size)


