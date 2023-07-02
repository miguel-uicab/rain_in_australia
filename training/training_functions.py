# !/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
# import plotly.express as px
# import plotly.figure_factory as ff
# import shap
import datetime
import logging
import yaml
from sklearn.impute import KNNImputer
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import jaccard_score, roc_auc_score, average_precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from category_encoders.count import CountEncoder

logging_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


# Funciones ###################################################################
###############################################################################
def get_config():
    """
    Se carga el archivo config.yaml.
    """
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def get_data(name_sav=None, path=None):
    """
    Retorna un DataFrame derivivado de un archivo csv.
    """
    name_sav = pickle.load(open(f'{path}{name_sav}', 'rb'))

    return name_sav


def make_dictionary(key_list=None, value_list=None):
    """
    Dados dos listas, una de llaves y otra de valores, se encarga de
    elaborar un diccionario. Si en los valores aparece 'np.nan', este es
    convertido a np.nan de numpy.
    """
    value_list_with_nan = [np.nan if x == 'np.nan' else x for x in value_list]
    dict_words_to_replace = dict(zip(key_list, value_list_with_nan))

    return dict_words_to_replace


def correct_name_keys(dictionary=None, word=None):
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


def hyper_space(model_name=None, random_state=None):
    """
    Selección del espacio hiperparametral dependiendo del
    modelo de machine learning.
    """
    if model_name == 'HistGradientBoostingClassifier':
        space = {'estimator__random_state': [random_state],
                 'estimator__max_iter': [100, 400],
                 'estimator__max_leaf_nodes': [10, 20],
                 'estimator__min_samples_leaf': [10, 20],
                 'estimator__learning_rate': [0.1, 0.03],
                 'estimator__max_depth': [None, 1],
                 'estimator__l2_regularization': [0]}
    elif model_name == 'RandomForestClassifier':
        space = {'estimator__random_state': [random_state],
                 'estimator__n_estimators': [100, 150],
                 'estimator__max_depth': [None, 1],
                 'estimator__min_samples_leaf': [1, 2],
                 'estimator__min_samples_split': [2, 3],
                 'estimator__max_leaf_nodes': [None, 2]}
    return space


def get_preprocessor(model_name=None, float_names=None,
                     categorical_names=None):
    """
    Selección de los procesos que irán en tuberías para que sean tomados
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
    numeric_transformer = Pipeline(
        steps=[('imputer', KNNImputer(n_neighbors=5, weights="uniform"))])
    categorical_transformer = Pipeline(
        steps=[('CountEncoder', CountEncoder(normalize=True))])
    if model_name == 'HistGradientBoostingClassifier':
        preprocessor = ColumnTransformer(remainder='passthrough',
                                         transformers=[('categorical', categorical_transformer, categorical_names)])
    else:
        preprocessor = ColumnTransformer(remainder='passthrough',
                                         transformers=[('numeric', numeric_transformer, float_names),
                                                       ('categorical', categorical_transformer, categorical_names)])

    return preprocessor


def get_feature_importances(model_name=None, feature_names_order=None,
                            clf_final=None, X_test=None, path=None,
                            objective_name=None, ratio_balance=None):
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
    Debido al uso de tuberías de procesos, el orden de las variables importa.
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


def optimization_model(name_sav=None,
                       features_names=None,
                       objective_name=None,
                       model_name=None,
                       seed=None,
                       ratio_balance=None,
                       test_size=None,
                       k_folds=None,
                       verbose=None,
                       optimized_metric=None,
                       save_best_info=False,
                       path=None):
    """
    Desarrolla una búsqueda de la mejor combinación de hiperparámetros,
    dado un estimador. Sus parámetros son:
    * original_data: Arvhivo .csv resultante del script "workshop.py".
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
    """
    logging.info('PROCESO DE OPTIMIZACIÓN.')
    # Configuración de data ##################################################
    logging.info('CONFIGURACIÓN DE DATA.')
    original_data = get_data(name_sav=name_sav, path=path)
    data = original_data[features_names]
    float_names = list(data.select_dtypes(include='float64').columns)
    categorical_names = list(data.select_dtypes(include='object').columns)
    int_names = list(data.select_dtypes(include='int').columns)
    feature_names_order = get_feature_names_order(model_name=model_name,
                                                  float_names=float_names,
                                                  categorical_names=categorical_names,
                                                  int_names=int_names)
    features = data[feature_names_order]
    label = original_data[objective_name]

    # Split data #############################################################
    logging.info('SPLIT DATA.')
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        label,
                                                        random_state=seed,
                                                        test_size=test_size,
                                                        stratify=label)

    # Configuración de Métricas ###############################################
    scores = {'f1': 'f1',
              'accuracy': 'accuracy',
              'balanced_accuracy': 'balanced_accuracy',
              'precision': 'precision',
              'recall': 'recall',
              'roc_auc': 'roc_auc',
              'average_precision': 'average_precision',
              'm_c': make_scorer(matthews_corrcoef),
              'jaccard': 'jaccard'}

    # Tuberías para Procesamiento y Muestreo ##################################
    logging.info('TUBERÍAS PARA PROCESAMIENTO Y MUESTREO.')
    preprocessor = get_preprocessor(model_name=model_name,
                                    float_names=float_names,
                                    categorical_names=categorical_names)
    estimator = select_model(model_name=model_name)
    transform = Pipeline(steps=[("processing", preprocessor),
                                ("RandomUnderSampler", RandomUnderSampler(
                                    random_state=seed, sampling_strategy=ratio_balance)),
                                ("estimator", estimator())])

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
                               'cv_accuracy': results['mean_test_accuracy'],
                               'cv_balanced_accuracy': results['mean_test_balanced_accuracy'],
                               'cv_precision': results['mean_test_precision'],
                               'cv_recall': results['mean_test_recall'],
                               'cv_roc_auc': results['mean_test_roc_auc'],
                               'cv_average_precision': results['mean_test_average_precision'],
                               'cv_m_c': results['mean_test_m_c'],
                               'cv_jaccard': results['mean_test_jaccard'],
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
    best_hyper['test_size'] = test_size

    if save_best_info:
        # Guardado de archivos #########################################
        name_best_info = f'best_hyper_info_{objective_name}_{ratio_balance}.sav'
        save_name = f'{path}{name_best_info}'
        logging.info('GUARDADO DE INFORMACIÓN DEL MEJOR MODELO.')
        pickle.dump(best_hyper, open(save_name, 'wb'))
    logging.info('FIN DE OPTIMIZACIÓN.')

    return best_hyper


def training_model(best_hyper_info=None,
                   name_sav=None,
                   features_names=None,
                   objective_name=None,
                   with_feature_importances=None,
                   with_probability_density=None,
                   save_data_train_test=None,
                   path=None):
    logging.info('PROCESO DE ENTRENAMIENTO Y GUARDADO DE MODELO.')
    # Configuración de data ###############################################
    logging.info('CONFIGURACIÓN DE DATA.')
    seed = best_hyper_info['hyperparameters']['random_state']
    ratio_balance = best_hyper_info['ratio_balance']
    model_name = best_hyper_info['model_name']
    test_size = best_hyper_info['test_size']
    original_data = get_data(name_sav=name_sav, path=path)
    data = original_data[features_names]
    float_names = list(data.select_dtypes(include='float64').columns)
    categorical_names = list(data.select_dtypes(include='object').columns)
    int_names = list(data.select_dtypes(include='int').columns)
    feature_names_order = get_feature_names_order(model_name=model_name,
                                                  float_names=float_names,
                                                  categorical_names=categorical_names,
                                                  int_names=int_names)
    features = data[feature_names_order]
    original_data['home_application_id']
    label = original_data[objective_name]

    # Split data ##########################################################
    logging.info('SPLIT DATA.')
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        label,
                                                        random_state=seed,
                                                        test_size=test_size,
                                                        stratify=label)

    # Tuberias para Procesamiento y Muestreo #############################
    logging.info('TUBERÍAS PARA PROCESAMIENTO Y MUESTREO.')
    preprocessor = get_preprocessor(model_name=model_name,
                                    float_names=float_names,
                                    categorical_names=categorical_names)
    estimator = select_model(model_name=model_name)
    transform = Pipeline(steps=[("processing", preprocessor),
                                ("RandomUnderSampler", RandomUnderSampler(
                                    random_state=seed, sampling_strategy=ratio_balance)),
                                ("estimator", estimator())])

    # Ajuste Final ###########################################################
    logging.info('AJUSTE FINAL.')
    best_param = best_hyper_info['hyperparameters']
    transform = Pipeline(steps=[("processing", preprocessor),
                                ("RandomUnderSampler", RandomUnderSampler(
                                    random_state=seed, sampling_strategy=ratio_balance)),
                                ("estimator", estimator(**best_param))])
    clf = transform.fit(X_train, y_train)
    clf.score(X_test, y_test)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    a_s = accuracy_score(y_test, y_pred)
    a_p_s = average_precision_score(y_test, y_pred)
    f_1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    m_c = matthews_corrcoef(y_test, y_pred)
    j_c = jaccard_score(y_test, y_pred)
    b_a = balanced_accuracy_score(y_test, y_pred)
    precision_maj = list(precision_score(y_test, y_pred, average=None))[0]
    recall_maj = list(recall_score(y_test, y_pred, average=None))[0]

    best_hyper_info['final_confusion_matrix'] = cm
    best_hyper_info['final_recall'] = recall
    best_hyper_info['final_precision'] = precision
    best_hyper_info['final_accuracy'] = a_s
    best_hyper_info['final_average_precision'] = a_p_s
    best_hyper_info['final_f1_score'] = f_1
    best_hyper_info['final_roc_auc'] = roc_auc
    best_hyper_info['final_matthews_corrcoef'] = m_c
    best_hyper_info['final_balanced_accuracy'] = b_a
    best_hyper_info['final_jaccard_score'] = j_c
    best_hyper_info['final_recall_maj'] = recall_maj
    best_hyper_info['final_precision_maj'] = precision_maj
    best_hyper_info['train_date'] = datetime.datetime.now()
    pd_best_info = pd.DataFrame(best_hyper_info).T
    save_name_best_info = f'final_model_info_{objective_name}_{ratio_balance}.sav'
    save_name = f'{path}{save_name_best_info}'
    pickle.dump(pd_best_info, open(save_name, 'wb'))
    pickle.dump(clf, open(f'{path}screening_model.sav', 'wb'))

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
                                                      clf_final=clf, X_test=X_test,
                                                      path=path, objective_name=objective_name,
                                                      ratio_balance=ratio_balance)
        name_feature_importances = f'final_feature_importances_{objective_name}_{ratio_balance}.sav'
        save_name = f'{path}{name_feature_importances}'
        pickle.dump(feature_importances, open(save_name, 'wb'))

    if save_data_train_test:
        logging.info('GUARDADO DE DATA TRAIN Y DATA TEST.')
        con_train = pd.concat([X_train, y_train], axis=1)
        merge_train = pd.merge(con_train, original_data['home_application_id'],
                               left_index=True, right_index=True, how="left")

        con_test = pd.concat([X_test, y_test], axis=1)
        merge_test = pd.merge(con_test, original_data['home_application_id'],
                              left_index=True, right_index=True, how="left")
        save_name_train = f'{path}data_train.sav'
        save_name_test = f'{path}data_test.sav'
        pickle.dump(merge_train, open(save_name_train, 'wb'))
        pickle.dump(merge_test, open(save_name_test, 'wb'))

    logging.info('FIN DE AJUSTE FINAL.')

    return best_hyper_info
