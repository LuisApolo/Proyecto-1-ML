# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn import preprocessing, utils
from sklearn.metrics import confusion_matrix


def load():
    #abrir archivo y asignar nombre a cada columna
    df = pd.read_csv("bupa.data", names=["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"])
    
    #borrar valores duplicados
    df = df.drop_duplicates()
    
    #asignar variable dependiente a una lista
    y = df['drinks']
    
    #dicotimizar variable dependiente 'drinks' > 3
    y = y.replace([0.5, 1.0, 2.0, 3.0], 0)
    y = y.replace([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 16.0, 20.0], 1)
    
    #borrar variable dependiente y variable selector
    df = df.drop('drinks', axis = 1)
    
    #llamar funcion para graficar matriz de correlacion
    plot_corr(df)
    X = df
    
    #Llamar a funcion de regresion logistica
    logistic_Regression(X, y)
    
def plot_corr(df):
    #matriz de correlación
    correlation = df.corr()
    plt.figure(figsize = (15, 10))
    sns.heatmap(correlation, annot = True, cmap = 'coolwarm')


def plot_cm(y_test, predictions):
    #matriz de confusion
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize = (3,3))
    sns.heatmap(cm, annot = True, cmap = 'Accent_r')
    
    
def logistic_Regression(X, y):
    #separando en train y test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    lg = LogisticRegression()
    
    #convertir float a int
    lab_enc = preprocessing.LabelEncoder()
    y_train_encoded = lab_enc.fit_transform(y_train)
    
    #predicciones
    lg.fit(X_train, y_train_encoded)
    predictions = lg.predict(X_test)
    print(predictions)

    #llamar funcion para graficar matriz confusion
    plot_cm(y_test, predictions)
    
    #exactitud
    score = lg.score(X_test, y_test)
    print(score)
        

def run():
    load()


if __name__=="__main__":
    run()
