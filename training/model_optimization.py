# !/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter("ignore")
from training_functions import *

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
model_name = 'HistGradientBoostingClassifier'
seed = 5000
ratio_balance = 1
k_folds = 4
verbose = 10
optimized_metric='cv_f1'
save_best_info=False
path='../outputs/'

optim_results = optimization_model(name_train_sav=name_train_sav,
                                   name_test_sav=name_test_sav,
                                   features_names=features_names,
                                   objective_name=objective_name,
                                   model_name=model_name,
                                   seed=seed,
                                   ratio_balance=ratio_balance,
                                   k_folds=k_folds,
                                   verbose=verbose,
                                   optimized_metric=optimized_metric,
                                   save_best_info=save_best_info,
                                   path=path)

pickle.dump(optim_results, open('../outputs/optimization_results.sav', 'wb'))
