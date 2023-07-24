# !/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI
from prediction_functions import *


# Initiate app instance ######################################################
app = FastAPI(title='Prediction Rain in Australia',
              version='1.0',
              description='Random forest model is used for prediction')


# Función de predicción ######################################################
@app.post("/predict")
def predict(json_dict: Json_Dict,
            path='../outputs/'):
    """ 
    Toma como entrada una estructura json con la features
    necesarias para la predicción.
    Devuelve una diccionario con las siguientes llaves:
    1. probability: Probabilidad de que llueva mañana.
    2. category: Categoría dependiente de la probabilidad.
                 Si la probabilidad es mayor a 0.5, la categoría será
                 'SI', en caso contrario, será 'NO'.
    3. version: Versión del modelo.
    """

    config = get_config()
    features_names = config['features_names']
    float_names = config['float_names']
    variable_names = config['variable_names']

    row_data = [json_dict.Location,
                json_dict.WindGustDir,
                json_dict.WindDir9am,
                json_dict.WindDir3pm,
                json_dict.RainToday,
                json_dict.MinTemp,
                json_dict.Rainfall,
                json_dict.WindGustSpeed,
                json_dict.WindSpeed3pm,
                json_dict.Humidity9am,
                json_dict.Humidity3pm,
                json_dict.Pressure9am,
                json_dict.Date
                ]

    df = pd.DataFrame(columns=variable_names,
                      data=np.array(row_data).reshape(1, 13))

    # Se convierte a tipó numérico las variables correspondientes.
    df = transform_data_type_to_float(data=df,
                                      list_names=float_names)

    # Se reconstruyen variables 'month' y 'week_of_year'.
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['month'] = df['month'].astype('object')
    df['week_of_year'] = df['week_of_year'].astype('object')

    # Se ordenan variables.
    data_features = df[features_names]
    float_names = list(data_features.select_dtypes(include='float64').columns)
    categorical_names = list(data_features.select_dtypes(include='object').columns)
    feature_names_order = get_feature_names_order(model_name='HistGradientBoostingClassifier',
                                                  float_names=float_names,
                                                  categorical_names=categorical_names)

    # Se imputa 'Unidentified' en los missing de variables categóricas.
    data_features = fillna_categoric_data(data=data_features,
                                          list_names=categorical_names)

    # Se tiene la data con el orden correcto de las variables.
    data_order = data_features[feature_names_order]

    # Se carga modelo.
    clf = pickle.load(open(f'{path}rain_model.sav', 'rb'))

    # Cálculo de probabilidades.
    predict_array = clf.predict_proba(data_order)
    probability = round(predict_array[0][1], 2)

    # Categoría.
    if probability <= 0.5:
        category = 'NO'
    else:
        category = 'SI'

    # Estructura json resultante.
    output = {'probability': probability,
              'category': category,
              'version': '1.0'}

    return output


if __name__ == "__main__":
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8000,
                log_level="info",
                reload=True)
