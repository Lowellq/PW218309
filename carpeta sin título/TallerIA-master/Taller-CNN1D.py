# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 09:32:11 2018
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
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential

#Se carga el dataset que se utilizara
dataset =  DataFrame(pd.read_csv('windserie.csv', header=None))

#Se define el tamaño de ventana que se utilizara pare el modelo
size = 50

#Se definen cantidad de salidas
outputs = 1

#Se define cantidad de epochs a entrenar el modelo

epochs = 50

#En este paso, se toma la serie de tiempos, se procesa para poder utilizarse
#en nuestro modelo de IA. Este proceso incluye normalización de datos entre 
#de 0 a 1.
dataset = tls.series_to_supervised(dataset,size, n_out=outputs, dropnan = True)
print(dataset)
datasetnorm = preprocessing.minmax_scale(dataset, feature_range=(0, 1))
print(datasetnorm)
datasetnorm = DataFrame(datasetnorm)
zy = datasetnorm.iloc[:, -1:]
zx = datasetnorm.iloc[:, :-1].values
X = np.array(zx, dtype='float64')
y = np.array(zy, dtype='float64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=20)

#Se cambia la forma de la colección de datos, para poder utilizarse con nuestro modelo.
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))

#Se define la topologia del modelo
model = Sequential((
        Convolution1D(4, 5,activation='relu',input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(),
        Convolution1D(4, 5,activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(outputs, activation='linear'),
        ))
#Se compila el modelo
model.compile(loss='mse', optimizer='adam', metrics=['mse','mape'])

#Se entrena modelo y se guarda información como "history"
history = model.fit(X_train, y_train, 24, epochs, validation_data=(X_test, y_test), verbose = 1)

#Se grafica el MSE
pyplot.style.use("ggplot")
pyplot.plot(history.history['mean_squared_error'], label='mse')
pyplot.plot(history.history['val_mean_squared_error'], label='val_mse')
pyplot.xlabel('Epochs')
pyplot.ylabel('Value')
pyplot.legend()
pyplot.show()
pyplot.plot(history.history["mean_absolute_percentage_error"], label="mape")
pyplot.plot(history.history["val_mean_absolute_percentage_error"], label="val_mape")
pyplot.xlabel('Epochs')
pyplot.ylabel('Value')
pyplot.legend()
pyplot.show()

#Predicciones con datos de prueba
predictions = model.predict(X_test, 10, verbose=2)

#Se grafican predicciones
pyplot.plot(y_test[1:100,], label='Real')
pyplot.plot(predictions[1:100,], label='Predicción')
pyplot.title('Valor predicción contra real')
pyplot.xlabel('Epochs')
pyplot.ylabel('Value')
pyplot.legend()
pyplot.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
pyplot.show()

