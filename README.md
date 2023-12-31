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
6. __app/:__ Contiene los archivos necesarios para poder usar la __*función de pronostico*__ a treavés de una app usando FastAPI.
7. __frontend/:__ Contiene los archivos necesarios para poder acceder a una visualización de la aplicación web que hace uso de la app usando Streamlit.
8. __images/:__ Contiene imagenes necesarias.


## __4. SOBRE EL MODELO DE MACHINE LEARNING.__

El modelo de ML asociado es uno que trata un problema de clasificación binaria. La variable a predecir es __RainTomorrow__, la cual sufre de un desbalanceo considerable entre sus clases mayoritaria ("NO") y minoritaria ("SI").

Para entender las decisiones y procesos llevados a cabo para el desarrollo del modelo de ML, así como las características y métricas finales asociadas al mismo, hay que consultar los jupyter noteboks contenidos en las diferentes carpetas. El orden en que deben consultarse estos archivos son:

1. __exploratory_and_preprocessing/exploratory_analysis_and_preprocessing.ipynb__: Se divide en dos partes (dos jupyter notebooks). En ellas se lleva a cabo un análisis exploratorio de datos además de una primera etapa de preprocesamiento. Entre los procesos que se realizan están: análisis de la correlación, tratamiento de valores perdidos, selección de variables y guardado de los binarios contenedores de las datas de entrenamiento y de testeo.

2. __training/model_selection.ipynb :__ En él se lleva a cabo una minicompetencia de modelos. Contiene importantes explicaciones de las consideraciones técnicas asociadas al entreamiento de un modelo de clasificación con un considerable desbalanceo entre sus clases.

3. __training/model_optimization.ipynb :__ Se generalizan prodecimientos llevados a cabo en la competencia de modelos y realiza la optimización del modelo ganador.
Tiene una versión en script llamada __model_optimization.py__, la cual es funcional pero aún falta por refinar. Este script puede ejecutarse de la siguiente manera desde consola:
```
python model_optimization.py
```

4. __training/final_training.ipynb :__ Se lleva a cabo el ajuste final del modelo optimizado y el guardado del archivo binario que lo contendrá.
Tiene una versión en script llamada __final_training.py__, la cual es funcional pero aún falta por refinar. Este script puede ejecutarse de la siguiente manera desde consola:
```
python final_training.py
```

Cada notebook tiene asociado un script de python que contiene las funciones necesarias que han sido construidas para no saturar el contenido de los notebooks. Cada una de estas funciones contiene una descripción de la labor que realizan.


## __5. SOBRE LA FUNCIÓN DE PRONÓSTICO.__
La __*función de pronóstico*__ está contenida en el script `prediction.py` en la carpeta __prediction/__. Esta función tiene como payload de entrada una estructura `json` de la siguiente forma
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
__prediction/uso_de_la_funcion_pronostico.ipynb__.


## __6. USO DE LA APP__

### Correr la app de manera local
Para hacer uso de la app de manera local, se puede hacer desde la carpeta __app/:__ y correr el siguiente código:

```
python python main.py
```

Para probarla, puede hacer uso de postman de la siguiente manera:
1. Ingresar el endpoint (`http://0.0.0.0:8000`) y seleccionar __POST__.

![pos1](images/app1.png)

2. Seleccionar __Body__, luego __raw__ y luego __JSON__. Ingresar un ejemplo de payload (uno se exhibe en la sección 5.)

![pos2](images/app2.png)

3. Seleccionar __SEND__.

![pos3](images/app3.png)

Se debe esperar recibir una respuesta del estilo siguiente:

![pos4](images/app4.png)

### Contenerización de la app

Para construir una imagen usando docker, se puede correr lo siguiente:
```
docker build -t rain_in_australia_app .   
```

Se puede correr la app, ahora ya en un contendor, usando lo siguiente:
```
docker run -p 8000:8000 rain_in_australia_app 
```

## __7. PROTOTIPO DE LA APLICACIÓN WEB__

### Para visualizar de maner local
Para tener una idea del frontend que servirá la app, previo a que esta ya haya sido corrida (ver sección 6.), se puede corres una aplicación Streamlit de la sigueinte manera.

```
streamlit run streamlit-app.py
```

Al hacerlo, se podrá ver un frontend como sigue:

![stream1](images/stream1.png)
![stream1](images/stream2.png)

### Contenerización de la app Streamlit

Para construir una imagen usando docker, se puede correr lo siguiente:
```
docker build -t rain_in_australia_streamlit_app . 
```

Se puede correr la app, ahora ya en un contendor, usando lo siguiente:
```
docker run -p 8501:8501 rain_in_australia_streamlit_app
```

## __8. FUTURO DEL PROYECTO: DESPLIEGUE DE LA APLICACIÓN Y MONITOREO.__

Queda pendiente la construcción de infraestructura necesaria para llegar a cumplir a cabalidad el objetivo último del proyecto: la escalabilidad de la __*función de pronóstico*__. Esta escalabilidad conlleva el despliegue del modelo y su monitoreo.

En cuanto al monitoreo, se refiera a tener bajo vigilancia los cambios que se tienen en la distribución de los datos a través del tiempo (__drift__), tanto en las features como en la variable objetivo. Cualquier cambio significativo debe dar lugar a una revisión del modelo, para su actualización o sustitución, pues este cambio puede afectar negativamente el rendimiento del modelo en producción. Para hacer esto, se puede recurrir a herramientas como [evently](https://github.com/evidentlyai/evidently).

El despliegue se puede hacer de manera rápida usando por ejemplo *AWS Elastic Beanstalk* para tener un endpoint que permita el consumo de la aplicacón de manera general. *AWS Elastic Beanstalk* permite desplegar usando solo un archivo `.zip` que contiene la aplicación y sus requerimientos.

Sin embargo, para añadir procedimientos de CI/CD (integración y despliegue  continuos) habrá que recurrir a una combinación más sofisticada de herramientas que permitan el dinamismo y versionado necesarios. Esto se puede alcanzar usando, por ejemplo:
1. *Docker*: para construir un contenedor de la api.
2. *Google Cloud Platform*: para guardar la imagen que permita la construcción del contenedor, correr la api y obtener un endpoint.
3. *GitHub Actions*: Para el entrenamiento continuo.
4. *DVC* (Data Version Control): Para guardar las diferentes versiones de la data de entrenamiento y del modelo que se tengan a lo largo del tiempo.
5. *CML* (Continuous Machine Learning): Para el monitoreo del modelo. 











