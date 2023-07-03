# __Rain in Australia__

## __1. OBJETIVO.__
El objetivo del proyecto es obtener un producto de Ciencia de Datos escalable que permita informar acerca de que si lloverá o no el día de mañana. Este pronóstico deberá basarse en datos, es decir, deberá basarse en información de data meteorológica histórica, además de considerar la localización en cierta ciudad australiana. La data se puede consular [aquí](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).

Para cumplir tal objetivo, se pretende desarrollar una api que pueda devolver, dada la fecha
del día en cuestión y ciertos datos meteorológicos y de localización, un string con la palabra "Sí" o "NO". El significado de estos strings es:
1. "SI": sí lloverá mañana.
2. "NO": no lloverá mañana.

## __2. MPV ACTUAL.__
Para esta etapa del proyecto, ha bastado con obtener una script de python con la __*función de pronóstico*__ asociada a la futura api. Es decir, el producto es exlusivamente técnico.

Esta función tiene como entrada una estructura `json` con la fecha e información meteorológica y de localización necesarias. (Para información técnica, consultar el apartado __5.__)

## __3. SOBRE EL REPOSITORIO EN GITHUB.__
Se puede consultar [aquí](https://github.com/miguel-uicab/rain_in_australia). El respositorio raíz se llama __rain_in_australia__. Contiene el desarrollo de código necesario para obtener la __*función de pronóstico*__, además de contener el preprocesamiento, análisis exploratorio, entrenamiento y optimización asociados al modelo de Machine Learning que está tras bambalinas de la __*función de pronóstico*__.

Para la codificación se ha usado python versión `3.8.12`.

El repositorio contiene 5 carpetas. Cada carpeta contiene un archivo *requiriments.txt* que contiene las versiones de paqueterías necesarias. Se recomienda instalar estas paqueterías en ambientes aislados.

Las 5 carpetas son:
1. __data/:__ Contiene un archivo con la data histórica necesaria para comenzar con el entrenamiento del modelo de machine learning.
2. __exploratory_and_preprocessing/:__ Contiene los archivos necesarios para el análsis exploratorio de datos y preprocesamiento.
3. __outputs/:__ Contiene los archivos binarios resultantes de los procesos de preprocesamiento de data y ajuste final modelo. Además, contiene gráficos informativos en formato `html`. De entre todos los archivos, el binario contenedor del modelo de ML es `rain_model.sav`.
4. __training/:__ Contiene los archivos necesarios para el entrenamiento, optimización, y guardado del modelo.
5. __prediction/:__ Contiene los archivos necesarios para poner en marcha la __*función de pronostico*__.

## __4. SOBRE EL MODELO DE MACHINE LEARNING.__

El modelo de ML asociado es uno que trata un problema de clasificación binaria. La variable a predecir es __RainTomorrow__, la cual sufre de un desbalanceo considerable entre sus clases mayoritaria ("NO") y minoritaria ("SI").

Para entender las decisiones y procesos llevados a cabo para el desarrollo del modelo de ML, así como las características y métricas finales asociadas al mismo, hay que consultar los jupyter noteboks contenidos en las diferentes carpetas. El orden en que deben consultarse estos archivos son:

1. __exploratory_and_preprocessing/exploratory_analysis_and_preprocessing.ipynb__: Se divide en dos partes (dos jupyter notebooks). En ellas se lleva a cabo un análisis exploratorio de datos además de una primera etapa de preprocesamiento. Entre los procesos que se realizan están: análisis de la correlación, tratamiento de valores perdidos, selección de variables y guardado de los binarios contenedores de las datas de entrenamiento y de prueba.

2. __training/model_selection.ipynb :__ En él se lleva a cabo una minicompetencia de modelos. Contiene importantes explicaciones de las consideraciones técnicas asociadas al entreamiento de un modelo de clasificación con un considerable desbalanceo entre sus clases.

3. __training/model_optimization.ipynb :__ Se generalizan prodecimientos llevados a cabo en la competencia de modelos y realiza la optimización del modelo ganador. 

4. __training/final_training.ipynb :__ Se lleva a cabo el ajuste final del modelo optimizado y el guardado del archivo binario que lo contendrá.

Cada notebook tiene asociado un script de python. Estos scripts contienen funciones necesarias que han sido construidas para no saturar el contenido de los notebooks. Cada una de estas funciones contiene una descripción de la labor que realizan.


## __5. SOBRE LA FUNCIÓN DE PRONÓSTICO.__
La __*función de pronóstico*__ está contenida en el script `prediction.py` en la carpeta __prediction/__. Esta función tiene como entrada una estructura `json` de la siguiente forma
```
{ "Location": "Albury",
  "WindGustDir": "ENE",
  "WindDir9am": null,
  "WindDir3pm": "ESE",
  "RainToday": "No",
  "MinTemp": 20.4,
  "Rainfall": 0.0,
  "WindGustSpeed": 54.0,
  "WindSpeed3pm": 7.0,
  "Humidity9am": 46.0,
  "Humidity3pm": 17.0,
  "Pressure9am": 1013.4,
  "Date": "2008-12-31"}
```
Con base en esta información preliminar, la función replica los procesos realizados en el preprocesamiento, las transformaciones
y la ingeniería de características, con el fin de obtener la lista definitiva de variables que son entradas del modelo de ML.

La salida de la función es una estructura `json` de la forma 
```
{'probability': 0.4, 'category': 'NO', 'version': '1.0'}
```
donde:
1. `probability`: Es la probabilidad de que llueva mañana.
2. `category`: Categoría dependiente de la probabilidad. Si la probabilidad es mayor a 0.5, la categoría será 'SI', en caso contrario, será 'NO'.
3. `version`: Es la versión del modelo.

Para saber cómo hacer uso de está función, hay que consultar el juputer notebook
__prediction/uso_de_la_funcion_prediccion.ipynb__.

## __6. FUTURO DEL PROYECTO: DESPLIEGUE Y MONITOREO.__

Queda pendiente la construcción de infraestructura necesaria para llegar a cumplir a cabalidad el objetivo último del proyecto: la escalabilidad de la __*función de pronóstico*__. Esta escalabilidad conlleva el despliegue del modelo y su monitoreo.

En cuanto al monitoreo, se refiera a tener bajo vigilancia los cambios que se tienen en la distribución de los datos a través del tiempo (__drift__), tanto en las features como en la variable objetivo. Cualquier cambio significativo debe dar lugar a una revisión del modelo, para su actualización o sustitución, pues este cambio puede afectar negativamente el rendimiento del modelo en producción. Para hacer esto, se puede recurrir a herramientas como [evently](https://github.com/evidentlyai/evidently).

El despliegue se puede hacer de manera rápida con lo siguiente. Usando frameworks como *FastAPI* con el fin de construir la api. Una vez hecho esto, se puede acceder a herramientas como *AWS Elastic Beanstalk* para tener un endpoint que permita el consumo de la aplicacón. *AWS Elastic Beanstalk* permite desplegar usando solo un archivo `.zip` que contiene la aplicación y sus requerimientos.

Sin embargo, para añadir procedimientos de CI/CD (integración y despliegue  continuos) habrá que recurrir a una combinación más sofisticada de herramientas que permitan el dinamismo y versionado necesarios. Esto se puede alcanzar usando, por ejemplo:
1. *Docker*: para construir un contenedor de la api.
2. *Google Cloud Platform*: para guardar la imagen que permita la construcción del contenedor, correr la api y obtener un endpoint.
3. *GitHub Actions*: Para el entrenamiento continuo.
4. *DVC* (Data Version Control): Para guardar las diferentes versiones de la data de entrenamiento y del modelo que se tengan a lo largo del tiempo.
5. *CML* (Continuous Machine Learning): Para el monitoreo del modelo. 











