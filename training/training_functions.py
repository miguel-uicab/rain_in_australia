# !/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import shap
import datetime
import logging
import yaml
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from category_encoders.count import CountEncoder

logging_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def get_config():
    """
    Se carga el archivo config.yaml.
    """
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def get_data(name_sav=None,
             path=None):
    """
    Retorna un DataFrame derivado de un archivo binario.
    """
    name_sav = pickle.load(open(f'{path}{name_sav}', 'rb'))

    return name_sav


def make_dictionary(key_list=None,
                    value_list=None):
    """
    Dados dos listas, una de llaves y otra de valores, se encarga de
    elaborar un diccionario. Si en los valores aparece 'np.nan', este es
    convertido a np.nan de numpy.
    """
    value_list_with_nan = [np.nan if x == 'np.nan' else x for x in value_list]
    dict_words_to_replace = dict(zip(key_list, value_list_with_nan))

    return dict_words_to_replace


def correct_name_keys(dictionary=None,
                      word=None):
    """
    Elimina palabras indeseables en los nombres de los hiperparámetros del
    modelo elegido, esto posterior al proceso de optimización del modelo.
    """
    keys = []
    values = []
    for key, value in dictionary.items():
        keys.append(key)
        values.append(value)

    keys = [s.replace('estimator__', '') for s in keys]
    new_dictionary = make_dictionary(key_list=keys, value_list=values)

    return new_dictionary


def select_model(model_name=None):
    """
    Selección del modelo de machine learning.
    """
    if model_name == 'LogisticRegression':
        estimator = LogisticRegression
    elif model_name == 'HistGradientBoostingClassifier':
        estimator = HistGradientBoostingClassifier
    elif model_name == 'RandomForestClassifier':
        estimator = RandomForestClassifier

    return estimator


def hyper_space(model_name=None,
                random_state=None):
    """
    Selección del espacio hiperparametral dependiendo del
    modelo de machine learning.
    """
    if model_name == 'HistGradientBoostingClassifier':
        space = {'estimator__random_state': [random_state],
                 'estimator__max_iter': [200, 400],
                 'estimator__max_leaf_nodes': [10], 
                 'estimator__min_samples_leaf': [10, 20],
                 'estimator__learning_rate':  [0.5],
                 'estimator__max_depth': [None, 1],
                 'estimator__l2_regularization': [0]}
        # space = {'estimator__random_state': [random_state],
        #          'estimator__max_iter': [200, 400, 650, 700, 800],
        #          'estimator__max_leaf_nodes': [10, 70, 100, 150], 
        #          'estimator__min_samples_leaf': [10, 20, 30, 40],
        #          'estimator__learning_rate':  [0.05, 0.07, 0.1, 0.3, 0.5, 1],
        #          'estimator__max_depth': [None, 1],
        #          'estimator__l2_regularization': [0]}
    elif model_name == 'RandomForestClassifier':
        space = {'estimator__random_state': [random_state],
                 'estimator__n_estimators': [100, 150],
                 'estimator__max_depth': [None, 1],
                 'estimator__min_samples_leaf': [1, 2],
                 'estimator__min_samples_split': [2, 3],
                 'estimator__max_leaf_nodes': [None, 2]}
    return space


def get_preprocessor(model_name=None,
                     float_names=None,
                     categorical_names=None):
    """
    Selección de los procesos que irán en Pipilines para que sean tomados
    en cuenta en los procesos de Cross-Validation. Estos procesos son pasados
    a variables Numéricas y Categóricas.
        1. Procesos para variables numéricas: Imputador. Este proceso no se
           aplica si el modelo de machine learning es
           HistGradientBoostingClassifier, puesto que este estimador toma
           en cuenta valores perdidos.
        2. Procesos para variables categóricas: Codificador.
    IMPORTANTE: los procesos aplicados a las variables hacen que estos
    cambien de posición con respecto a la data. Las "variables tranformadas"
    van quedando en la izquierda de la data mientras que las que no se
    transforman se van corriendo a la derecha.
    Ver función "get_feature_names_order" para saber cómo queda el
    orden de estas variables de acuerdo el modelo de machine learning elegido.
    """
    # numeric_transformer = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=3, weights="uniform"))])
    numeric_transformer = Pipeline(steps=[('imputer',
                                           SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[('CountEncoder',
                                               CountEncoder(normalize=True))])
    if model_name == 'HistGradientBoostingClassifier':
        preprocessor = ColumnTransformer(remainder='passthrough',
                                         transformers=[('categorical',
                                                        categorical_transformer,
                                                        categorical_names)])
    else:
        preprocessor = ColumnTransformer(remainder='passthrough',
                                         transformers=[('numeric',
                                                        numeric_transformer,
                                                        float_names),
                                                       ('categorical',
                                                        categorical_transformer,
                                                        categorical_names)])

    return preprocessor


def get_feature_importances(model_name=None,
                            feature_names_order=None,
                            clf_final=None,
                            X_test=None,
                            path=None,
                            objective_name=None,
                            ratio_balance=None):
    """
    Cálcula las importancias de las variables después del ajuste del modelo.
    1. Para modelo HistGradientBoostingClassifier: se usa shap-values.
    2. Otro modelo: feature importances.
    """
    if model_name == 'HistGradientBoostingClassifier':
        x_Test = clf_final.named_steps['processing'].transform(X_test)
        explainer = shap.Explainer(
            clf_final.named_steps["estimator"].predict_proba, x_Test)
        shap_test = explainer(x_Test)
        len_features = shap_test.values[0].shape[0]
        len_X_test = X_test.shape[0]
        sp_values_list = []
        for j in range(0, len_X_test):
            ind_sp_values_list = []
            for i in range(0, len_features):
                ind_sp_values_list.append(shap_test.values[j][i][1])
            sp_values_list.append(ind_sp_values_list)
            pd_shape = pd.DataFrame(
                sp_values_list, columns=list(X_test.columns))
            pd_f_i = abs(pd_shape).mean().to_frame()
            pd_f_i.reset_index(drop=False, inplace=True)
            pd_f_i.columns = ['feature_names', 'feature_importances']
        pickle.dump(x_Test, open(f'{path}x_Test.sav', 'wb'))
        pickle.dump(explainer, open(f'{path}explainer.sav', 'wb'))
        pickle.dump(shap_test, open(f'{path}shap_test.sav', 'wb'))
    else:
        feature_importances = list(
            clf_final._final_estimator.feature_importances_)
        pd_f_i = pd.DataFrame({'feature_names': feature_names_order,
                               'feature_importances': feature_importances})
    pd_f_i.sort_values(by='feature_importances',
                       ascending=False,
                       inplace=True)
    fig = px.bar(pd_f_i,
                 x='feature_names',
                 y='feature_importances',
                 color='feature_importances')
    fig.show()
    save_name_importances = f'final_feature_importances_{objective_name}_{ratio_balance}'
    fig.write_html(f'{path}{save_name_importances}.html')

    return pd_f_i


def density_prob(probabilities=None,
                 name_probability=None,
                 objective_name=None,
                 ratio_balance=None,
                 bin_size=None,
                 path=None):
    """
    Construcción de densidad de probabilidad. Se usa paquetería "plotly".
    """
    hist_data = [probabilities]
    group_labels = [name_probability]
    fig = ff.create_distplot(hist_data, group_labels,  bin_size=bin_size)
    fig.show()
    save_name_density = f'final_density_{objective_name}_{ratio_balance}'
    fig.write_html(f'{path}{save_name_density}.html')


def get_feature_names_order(model_name=None,
                            float_names=None,
                            categorical_names=None):
    """
    El uso de Pipilines implica que el orden de las variables importa.
    Este orden en las variables esta vínculado al orden en que suceden los
    procesos en las tuberías (ver función "get_preprocessor")
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


def optimization_model(name_train_sav=None,
                       name_test_sav=None,
                       features_names=None,
                       objective_name=None,
                       model_name=None,
                       seed=None,
                       ratio_balance=None,
                       k_folds=None,
                       verbose=None,
                       optimized_metric=None,
                       save_best_info=False,
                       path=None):
    """
    Desarrolla una búsqueda de la mejor combinación de hiperparámetros,
    dado un estimador. Sus parámetros son:
    * name_train_sav: Nombre del archivo .sav de entrenamiento.
    * name_test_sav: Nombre del archivo .sav de prueba.
    * features_names: Lista de nombres de features a considerar.
    * model_name: String. Selección del nombre del estimador.
      Por el momento solo hay dos opciones, 'HistGradientBoostingClassifier'
      y 'RandomForestClassifier'.
    * seed: Entero. Semilla aleatoria que gobernará los procesos.
    * ratio_balance: Decimal. El valor 1 permite tener una muestra cuya
      configuración es 50 % - 50 % con respecto a las
      clases mayoritaria y minoritaria. En caso de otro valor, es el porcentaje
      que n_minoritaria representa con respecto a n_mayoritaria, es decir,
      (n_mayoritaria) * (ratio_balance) = n_minoritaria.
    * k_folds: Entero. Número de folds de CV.
    * verbose: Entero. Mientras mayor sea, más información acerca del proceso
      de búsqueda (optimización) se arroja en forma de loggins.
    * optimized_metric: String. Métrica usada para localizar el mejor modelo.
      Por defualt se tiene 'cv_f1'.
    * save_best_info: Booleano. Si o no se guarda un archivo .sav que contiene
      la información de la mejor combinación de hiperparámetros.
    * path: String. Ubicación de los archivos de data.
    OUTPUT: Panda-series con la información relevante del modelo que ha optimizado
    la métrica dada por optimized_metric.
    """
    logging.info('PROCESO DE OPTIMIZACIÓN.')
    df_train = get_data(name_sav=name_train_sav, path=path)
    df_test = get_data(name_sav=name_test_sav, path=path)


    # Configuraciones Generales ##################################################
    logging.info('CONFIGURACIÓN GENERALES.')
    # Se obtiene el dataset de features en el conjunto de Entrenamiento. 
    original_data = df_train.copy()
    data_features = original_data[features_names]

    # Se obtienen los nombres de variables numéricas y de las categóricas.
    float_names = list(data_features.select_dtypes(include='float64').columns)
    categorical_names = list(data_features.select_dtypes(include='object').columns)
    # int_names = list(data.select_dtypes(include='int').columns)

    # Configuración de scores.
    scores = {'f1': 'f1',
              'precision': 'precision',
              'recall': 'recall',
              'm_c': make_scorer(matthews_corrcoef)}

    # Se ordenan los nombres de variables.
    feature_names_order = get_feature_names_order(model_name=model_name,
                                                  float_names=float_names,
                                                  categorical_names=categorical_names)

    # Split data #############################################################
    X_train = data_features[feature_names_order]
    y_train = original_data[objective_name]
    X_test = df_test[feature_names_order]
    y_test = df_test[objective_name]

    # Tuberías para Procesamiento y Muestreo ##################################
    logging.info('TUBERÍAS PARA PROCESAMIENTO Y MUESTREO.')
    preprocessor = get_preprocessor(model_name=model_name,
                                    float_names=float_names,
                                    categorical_names=categorical_names)
    estimator = select_model(model_name=model_name)
    transform = Pipeline(steps=[('processing', 
                                  preprocessor),
                                ('RandomUnderSampler',
                                  RandomUnderSampler(random_state=seed,
                                                     sampling_strategy=ratio_balance)),
                                ('estimator',
                                  estimator())])

    # Configuración para CV y Search #####################################
    logging.info('CONFIGURACIÓN PARA CV Y SEARCH.')
    param_grid = hyper_space(model_name=model_name, random_state=seed)
    c_v = StratifiedKFold(n_splits=k_folds,
                          shuffle=True,
                          random_state=seed)
    CV_model = GridSearchCV(estimator=transform,
                            param_grid=param_grid,
                            cv=c_v,
                            scoring=scores,
                            verbose=verbose,
                            refit=False)
    logging.info('COMIENZA OPTIMIZACIÓN.')
    CV_model.fit(X_train, y_train)

    # Cálcuo de métricas de CV #########################################
    logging.info('CÁLCULO DE MÉTRICAS DE CV.')
    results = CV_model.cv_results_
    pd_results = pd.DataFrame({'hyperparameters': results['params'],
                               'cv_f1': results['mean_test_f1'],
                               'cv_precision': results['mean_test_precision'],
                               'cv_recall': results['mean_test_recall'],
                               'cv_m_c': results['mean_test_m_c'],
                               'optimization_date':  datetime.datetime.now()})
    pd_results['ratio_balance'] = ratio_balance

    # Selección del mejor modelo vía una métrica de cv #######################
    logging.info('SELECCIÓN DEL MEJOR MODELO VÍA UNA MÉTRICA DE CV.')
    pd_results.sort_values(by=optimized_metric, ascending=False, inplace=True)
    pd_results.reset_index(drop=True, inplace=True)
    pd_results['hyperparameters'] = pd_results.apply(lambda x: correct_name_keys(dictionary=x['hyperparameters'], word='estimator__'),
                                                     axis=1)
    best_hyper = pd_results.loc[0, :]
    best_hyper['model_name'] = model_name
    best_hyper['k_folds'] = k_folds

    if save_best_info:
        # Guardado de archivos #########################################
        name_best_info = f'best_hyper_info_{objective_name}_{ratio_balance}.sav'
        save_name = f'{path}{name_best_info}'
        logging.info('GUARDADO DE INFORMACIÓN DEL MEJOR MODELO.')
        pickle.dump(best_hyper, open(save_name, 'wb'))
    logging.info('FIN DE OPTIMIZACIÓN.')

    return best_hyper


def training_model(best_hyper_info=None,
                   name_train_sav=None,
                   name_test_sav=None,
                   features_names=None,
                   objective_name=None,
                   with_feature_importances=None,
                   with_probability_density=None,
                   path=None):
    """
    Ajusta y guarda el modelo optimizado. Sus parámetros son:
    * best_hyper_info: Es output derivado de la funcion optimization_model.
    * name_train_sav: Nombre del archivo .sav de entrenamiento.
    * name_test_sav: Nombre del archivo .sav de prueba.
    * features_names: Lista de nombres de features a considerar.
    * model_name: String. Selección del nombre del estimador.
      Por el momento solo hay dos opciones, 'HistGradientBoostingClassifier'
      y 'RandomForestClassifier'.
    * with_feature_importances: Binario. Calcula la feature importances
      vía shap-values. Construye el gráfico correspondiente.
    * with_probability_density: Binario. Cálcula las probabilidades
      predichas en el conjunto de testeo. Construye el gráfico correspondiente
    * path: String. Ubicación de los archivos de data.
    OUTPUT: Panda-series con la información relevante del modelo optimizado.
    Se informan métricas de cross-validation y de testeo.
    """  
    logging.info('PROCESO DE ENTRENAMIENTO Y GUARDADO DE MODELO.')
    # Configuraciones Generales ###############################################
    logging.info('CONFIGURACIÓN DE DATA.')
    df_train = get_data(name_sav=name_train_sav, path=path)
    df_test = get_data(name_sav=name_test_sav, path=path)   
    seed = best_hyper_info['hyperparameters']['random_state']
    ratio_balance = best_hyper_info['ratio_balance']
    model_name = best_hyper_info['model_name']

    # Se obtiene el dataset de features en el conjunto de Entrenamiento. 
    original_data = df_train.copy()
    data_features = original_data[features_names]

    # Se obtienen los nombres de variables numéricas y de las categóricas.
    float_names = list(data_features.select_dtypes(include='float64').columns)
    categorical_names = list(data_features.select_dtypes(include='object').columns)
    # int_names = list(data.select_dtypes(include='int').columns)

    # Configuración de scores.
    scores = {'f1': 'f1',
              'precision': 'precision',
              'recall': 'recall',
              'm_c': make_scorer(matthews_corrcoef)}

    # Se ordenan los nombres de variables.
    feature_names_order = get_feature_names_order(model_name=model_name,
                                                  float_names=float_names,
                                                  categorical_names=categorical_names)

    # Split data #############################################################
    X_train= data_features[feature_names_order]
    y_train = original_data[objective_name]
    X_test = df_test[feature_names_order]
    y_test = df_test[objective_name] 

    # Tuberias para Procesamiento  #############################
    logging.info('TUBERÍAS PARA PROCESAMIENTO Y MUESTREO.')
    preprocessor = get_preprocessor(model_name=model_name,
                                    float_names=float_names,
                                    categorical_names=categorical_names)
    estimator = select_model(model_name=model_name)

    # Ajuste Final ###########################################################
    logging.info('AJUSTE FINAL.')
    best_param = best_hyper_info['hyperparameters']
    transform = Pipeline(steps=[("processing", preprocessor),
                                ("RandomUnderSampler", RandomUnderSampler(random_state=seed,
                                                                          sampling_strategy=ratio_balance)),
                                ("estimator", estimator(**best_param))])
    clf = transform.fit(X_train, y_train)
    clf.score(X_test, y_test)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    a_s = accuracy_score(y_test, y_pred)
    f_1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    m_c = matthews_corrcoef(y_test, y_pred)
    precision_maj = list(precision_score(y_test, y_pred, average=None))[0]
    recall_maj = list(recall_score(y_test, y_pred, average=None))[0]

    best_hyper_info['final_confusion_matrix'] = cm
    best_hyper_info['final_recall'] = recall
    best_hyper_info['final_precision'] = precision
    best_hyper_info['final_accuracy'] = a_s
    best_hyper_info['final_f1_score'] = f_1
    best_hyper_info['final_roc_auc'] = roc_auc
    best_hyper_info['final_matthews_corrcoef'] = m_c
    best_hyper_info['final_recall_maj'] = recall_maj
    best_hyper_info['final_precision_maj'] = precision_maj
    best_hyper_info['train_date'] = datetime.datetime.now()
    pd_best_info = pd.DataFrame(best_hyper_info).T
    save_name_best_info = f'final_model_info_{objective_name}_{ratio_balance}.sav'
    save_name = f'{path}{save_name_best_info}'
    pickle.dump(pd_best_info, open(save_name, 'wb'))
    pickle.dump(clf, open(f'{path}rain_model.sav', 'wb'))
    pickle.dump(X_test, open(f'{path}X_test.sav', 'wb'))
    pickle.dump(y_test, open(f'{path}y_test.sav', 'wb'))
    pickle.dump(X_train, open(f'{path}X_train.sav', 'wb'))
    pickle.dump(y_train, open(f'{path}y_train.sav', 'wb'))

    if with_probability_density:
        logging.info('CONSTRUCCIÓN DE DENSIDAD DE PROBABILIDADES DE X_TEST.')
        y_proba = clf.predict_proba(X_test)[:, 1]
        density_prob(probabilities=y_proba,
                     name_probability='Probabilidades de X_test',
                     objective_name=objective_name,
                     ratio_balance=ratio_balance,
                     bin_size=0.02,
                     path=path)

    if with_feature_importances:
        logging.info('OBTENCIÓN DE FEATURE IMPORTANCES.')
        feature_importances = get_feature_importances(model_name=model_name,
                                                      feature_names_order=feature_names_order,
                                                      clf_final=clf,
                                                      X_test=X_test,
                                                      path=path,
                                                      objective_name=objective_name,
                                                      ratio_balance=ratio_balance)
        name_feature_importances = f'final_feature_importances_{objective_name}_{ratio_balance}.sav'
        save_name = f'{path}{name_feature_importances}'
        pickle.dump(feature_importances, open(save_name, 'wb'))

    logging.info('FIN DE AJUSTE FINAL.')

    return best_hyper_info


def get_calibration_curve(y_test,
                          probs_test):
    """
    Se construye curva de calibración.
    """
    fop, mpv = calibration_curve(y_test,
                                 probs_test,
                                 n_bins=50,
                                 normalize=True)
    fig = go.Figure(data=go.Scatter(x=mpv,
                                    y=fop,
                                    mode='lines+markers'))
    fig.add_shape(type='line',
                  line=dict(dash='dash'),
                  x0=0,
                  x1=1,
                  y0=0,
                  y1=1)
    fig.update_layout(title='Calibration Test',
                      xaxis_title='Mean Predicted Probability in each bin',
                      yaxis_title='Ratio of positives')
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_traces(marker=dict(size=5,
                                  line=dict(width=2,
                                            color='gray')),
                      line=dict(width=1,
                                color='orange'))
    fig.update_traces(marker=dict(size=6, color='gray'),
                      line=dict(width=1,
                                color='orange'))
    fig.show()
