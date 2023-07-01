import pandas as pd
import numpy as np


def get_count_missing(data=None, feature_names=None):
    """
    Arroja un DataFrame con el conteo y el porcentaje de
    valores perdidos de cada variable float.
    """
    count_missing = pd.DataFrame(data[feature_names].isna().sum())
    count_missing.reset_index(drop=False, inplace=True)
    count_missing.columns = ['name', 'conteo']
    count_missing['porcentaje'] = round((count_missing['conteo']/data.shape[0])*100, 2)
    count_missing.sort_values(by='porcentaje', ascending=False, inplace=True)
    count_missing.reset_index(drop=True, inplace=True)
    
    return count_missing


def remove_features_with_missing(count=None, threshold=None):
    """
    Devuelve una lista de variables cuyos porcentajes de valores perdidos
    no supera cierto umbral.
    Depende del resultado de la función get_count_missing.
    """
    drop_by_missing = list(count_missing[count_missing['porcentaje']>=threshold]['name'])
    float_names_without_missing = total_float_names.copy()
    for name in drop_by_missing:
        float_names_without_missing.remove(name)
        
    return float_names_without_missing


def data_filter_by_upper_quantile(data=None,
                                  list_float_names=None,
                                  upper_quantile=0.99):
    """
    Elimina outliers de cierta lista de variables float.
    Aquí, los outliers son aquellos valores que sobrepasan cierto umbral de cuantil superior.
    """
    data_without_outliers = data.copy()
    for name in list_float_names:
        data_without_outliers = data_without_outliers[(data_without_outliers[name] <= np.nanquantile(data_without_outliers[name],
                                                       upper_quantile)) | (data_without_outliers[name].isna())]
    
    return data_without_outliers


def get_features_names_drop_by_corr(data=None,
                                    list_feature_names=None,
                                    threshold=None):
    """
    Devuelve una lista de variables que sobrepasan cierto valor de correlación
    con respecto a otras variables. Aquí, se ha usado la correlación no paramétrica
    de Spearman.
    """
    corr_matrix = (data[list_feature_names].corr(method='spearman').abs())
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
    drop_by_corr = [column for column in upper.columns if any(upper[column] > threshold)]

    return drop_by_corr


def fillna_categoric_data(data=None, list_names=None):
    """
    Dada una lista de variables categóricas, imputa, en los
    valores perdidos, el valor de 'No identificado'.
    """
    data_copy = data.copy()
    for name in list_names:
        data_copy[name].fillna('Unidentified', inplace=True)

    return data_copy