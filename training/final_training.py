# !/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter("ignore")
from training_functions import *

path='../outputs/'
optim_results = get_data(name_sav='optimization_results.sav', path=path)

with_feature_importances=True
with_probability_density=False
name_train_sav='wA_train.sav'
name_test_sav='wA_test.sav'
features_names =  ['Location',
                   'MinTemp',
                   'Rainfall',
                   'WindGustDir',
                   'WindGustSpeed',
                   'WindDir9am',
                   'WindDir3pm',
                   'WindSpeed3pm',
                   'Humidity9am',
                   'Humidity3pm', 
                   'Pressure9am',
                   'RainToday', 
                   'month',
                   'week_of_year']
objective_name = 'RainTomorrow'

train_results = training_model(best_hyper_info=optim_results,
                               name_train_sav=name_train_sav,
                               name_test_sav=name_test_sav,
                               features_names=features_names,
                               objective_name=objective_name,
                               with_feature_importances=with_feature_importances,
                               with_probability_density=with_probability_density,
                               path=path)