# !/usr/bin/env python
# coding: utf-8

import yaml
import pandas as pd
from pydantic import BaseModel
from typing import Optional


#############################################################################
class Json_Dict(BaseModel):
    """
    Variables needed for prediction.
    """
    Location: Optional[str] = None
    WindGustDir: Optional[str] = None
    WindDir9am: Optional[str] = None
    WindDir3pm: Optional[str] = None
    RainToday: Optional[str] = None
    MinTemp: Optional[float] = None
    Rainfall: Optional[float] = None
    WindGustSpeed: Optional[float] = None
    WindSpeed3pm: Optional[float] = None
    Humidity9am: Optional[float] = None
    Humidity3pm: Optional[float] = None
    Pressure9am: Optional[float] = None
    Date: Optional[str] = None


def get_config():
    """
    Se carga el archivo config.yaml.
    """
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def get_dataframe(json=None):
    """
    Convierte una estructura json con la informarción básica payload
    en un DataFrame. El .T rota el DataFrame con el fin de obtener
    información tipo fila.
    """
    df_json = pd.DataFrame.from_dict(json, orient='index').T

    return df_json


def fillna_categoric_data(data=None,
                          list_names=None):
    """
    Dada una lista de variables categóricas, imputa, en los
    valores perdidos, el valor de 'No identificado'.
    """
    data_copy = data.copy()
    for name in list_names:
        data_copy[name].fillna('Unidentified', inplace=True)

    return data_copy


def get_feature_names_order(model_name=None,
                            float_names=None,
                            categorical_names=None):
    """
    El uso de Pipilines implica que el orden de las variables importa.
    Este orden en las variables esta vínculado al orden en que suceden los
    procesos en las tuberías (ver función "get_preprocessor"
    de training_functions.py)
    * Orden de variables en el modelo HistGradientBoostingClassifier:
        1. Tipo categórica.
        2. Tipo float.
        3. Tipo enteras.
    * Orden de variables en otros modelos:
        1. Tipo float
        2. Tipo categórica.
        3. Tipo enteras.
    """
    if model_name == 'HistGradientBoostingClassifier':
        feature_names_order = categorical_names+float_names
        # feature_names_order = categorical_names+float_names+int_names
    else:
        feature_names_order = float_names+categorical_names
        # feature_names_order = float_names+categorical_names+int_names

    return feature_names_order


def transform_data_type_to_float(data=None,
                                 list_names=None):
    """
    Dada una lista de variables de interés, se tranforman
    a tipo float.
    """
    data_copy = data.copy()
    for name in list_names:
        data_copy[name] = data_copy[name].astype(float)

    return data_copy
