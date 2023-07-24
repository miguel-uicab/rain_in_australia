# !/usr/bin/env python
# coding: utf-8

import logging
logging_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(format=logging_format, datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
from prediction_functions import *


def prediction(json_dict=None,
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
    if json_dict is None:
        logging.info('NO SE HA RECIBIDO INFORMACIÓN ALGUNA.')
        return None
    else:
        logging.info('COMIENZO DE PREDICCIÓN.')

        # Se cargan nombres de features y de variables numéricas.
        config = get_config()
        features_names = config['features_names']
        float_names = config['float_names']

        # Se convierte json a dataframe.
        df = get_dataframe(json=json_dict)

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

        logging.info('FIN DE PREDICCIÓN.')
        return output
