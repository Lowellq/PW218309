# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 02:20:00 2018
https://gitlab.com/NightValbarz/TallerIA.git
@author: Night
"""

import numpy as np
import pandas as pd
import Tools as tls
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

#Se carga el dataset que se utilizara
dataset =  DataFrame(pd.read_csv('windserie.csv', header=None))

#Se define el tamaño de ventana que se utilizara pare el modelo
size = 24

#En este paso, se toma la serie de tiempos, se procesa para poder utilizarse
#en nuestro modelo de IA. Este proceso incluye normalización de datos entre 
#de 0 a 1.
dataset = tls.series_to_supervised(dataset,n_in=5, n_out=1, dropnan = True)
datasetnorm = preprocessing.minmax_scale(dataset, feature_range=(0, 1))
datasetnorm = DataFrame(datasetnorm)
zy = datasetnorm.iloc[:, -1:]
zx = datasetnorm.iloc[:, :-1].values
X = np.array(zx, dtype='float64')
y = np.array(zy, dtype='float64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=20)


#Se define nuestro modelo, con un numero de "vecinos" deseados a utilizarse.
knn = KNeighborsRegressor(n_neighbors=12)

#Se entrena el modelo
knn.fit(X_train, y_train)

#Se utiliza el modelo entrenado para realizar predicciones
pred = knn.predict(X_test)

#Se obtiene el error cuadratico medio comparando predicciones con valores reales
MSE = mean_squared_error(y_test, pred)
print ('MSE '+ str(MSE))

#Se imprime grafica
pyplot.style.use("ggplot")
pyplot.plot(y_test[1:100,]) #Valores ajustables para grafica
pyplot.plot(pred[1:100,])
pyplot.xlabel('Epochs')
pyplot.ylabel('Value')
pyplot.title('Predicción contra Real')
pyplot.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
pyplot.show()


#En este paso, realizaremos entrenamiento con multiples valores de vecinos, para
#buscar un parametro de vecinos que nos de resultado con mejor aptitud.
neighbors = 150
MSEList = []
for k in range(1,neighbors):
    knn = KNeighborsRegressor(k)
    knn.fit(X_train, y_train)
    MSEList.append(mean_squared_error(y_test, knn.predict(X_test)))

#Se grafica comportamiento de vecinos, donde se puede observar el mejor valor de K

pyplot.plot(range(1,neighbors), MSEList)
pyplot.xlabel('k-neighbors')
pyplot.ylabel('Mean Squared Error')
pyplot.title('Gráfica parametro k-Vecinos')
pyplot.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
pyplot.show()