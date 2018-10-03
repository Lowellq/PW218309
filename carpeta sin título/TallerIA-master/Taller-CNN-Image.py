# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 12:18:13 2018
https://gitlab.com/NightValbarz/TallerIA.git
@author: Night
"""

import numpy as np
import pandas as pd
import Tools as tls
from matplotlib import pyplot
from matplotlib import image
from pandas import DataFrame
from keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential



#Metodo usado para construcción del modelo
def buildModel(filt, kernel, dropout, in_shape):
    model = Sequential((
            Convolution2D(filters=filt,kernel_size=kernel, activation='relu',input_shape=in_shape),
            MaxPooling2D(pool_size=(3, 3)),
            Dropout(dropout),
            Convolution2D(filters=filt,kernel_size=kernel, activation='relu'),
            Convolution2D(filters=filt,kernel_size=kernel, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(filters=filt,kernel_size=kernel, activation='relu'),
            Convolution2D(filters=filt,kernel_size=kernel, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.25),
            Dense(1,activation='linear'),
            ))
            
    model.compile(loss='mse', optimizer='adam', metrics=['mape','mse'])
    return model
    
#Metodo principal
def main():
    
    #Se carga el dataset
    dataset =  DataFrame(pd.read_csv('windserie.csv', header=None))
    
    #Se define el tamaño de imagen a utilizarse
    IMAGE_SIZE=64
    
    #Se define epochs de entrenamiento
    epochs = 2
    
    #Se define el batch size, o "lote" de datos a usarse por paso
    
    BATCH_S=24
    
    #Se envia las series de tiempo, y el temaño de imagen, el metodo nos regresa
    #de vuelta los datos la colección de datos de las series de tiempo en matrices.
    X_train, y_train, X_test, y_test = tls.SerieToImage(dataset, IMAGE_SIZE)
    
    #Se convierten las listas a numpy array
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    
    #Se cambian la forma de los numpy array para el modelo
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],X_test.shape[2], 1))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    
    #Se envian matrices de datos de series de tiempo para generar las imagenes en formato jepg.
    #Se guardan en el directiorio definido.
    tls.imageDataToJPG(X_train, y_train, 'images/train')
    tls.imageDataToJPG(X_test, y_test, 'images/test')
    
    #Se carga y muestra una imagen creada como ejemplo, para observacion de serie de tiempo como imagen
    print('Ejemplo serie de tiempo como imagen')
    img = image.imread('images/train/13.jpeg')
    pyplot.imshow(img)
    pyplot.show()
    
    #Se construye modelo con parametros definidos de: 8 filtros, 3 tamaño kernel, 0.25 dropout, e input de 64,64,1
    model = buildModel(8, 3, 0.05, (IMAGE_SIZE,IMAGE_SIZE,1))
    
    #Se entrena modelo y se guarda historia.
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_S, validation_data=(X_test, y_test), verbose = 1)
    
    pyplot.style.use("ggplot")
    pyplot.figure()
    N = epochs
    pyplot.plot(np.arange(0, N), history.history["mean_squared_error"], label="mse")
    pyplot.plot(np.arange(0, N), history.history["val_mean_squared_error"], label="val_mse")
    pyplot.title("Training MSE")
    pyplot.xlabel("Epoch #")
    pyplot.ylabel("MSE")
    pyplot.legend(loc="upper right")
    pyplot.show()
    pyplot.plot(np.arange(0, N), history.history["mean_absolute_percentage_error"], label="mape")
    pyplot.plot(np.arange(0, N), history.history["val_mean_absolute_percentage_error"], label="val_mape")
    pyplot.title("Training MAPE")
    pyplot.xlabel("Epoch #")
    pyplot.ylabel("MAPE")
    pyplot.legend(loc="upper right")
    pyplot.show()
    
    #Se realizan predicciones con datos de prueba
    predictions = model.predict(X_test, BATCH_S, verbose=2)
    
    #Se grafican predicciones
    pyplot.plot(y_test[1:100,], label='Real')
    pyplot.plot(predictions[1:100,], label='Predicción')
    pyplot.title('Valor predicción contra real')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Value')
    pyplot.legend()
    pyplot.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    pyplot.show()
    
if __name__ == '__main__':
    main()