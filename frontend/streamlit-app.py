import streamlit
import requests
import json
import pandas as pd

df = pd.read_csv("weatherAUS.csv")
df.dropna(subset=['RainTomorrow', 'RainToday'],
          inplace=True)


def run():
    """
    Desarrollo de un prototipo de aplicación web que despliega un modelo de
    predicción de lluvia.
    """

    streamlit.title("Rain in Australia Prediction")
    Location = streamlit.selectbox("Location", df.Location.unique())
    WindGustDir = streamlit.selectbox("WindGustDir", df.WindGustDir.unique())
    WindDir9am = streamlit.selectbox("WindDir9am", df.WindDir9am.unique())
    WindDir3pm = streamlit.selectbox("WindDir3pm", df.WindDir3pm.unique())
    RainToday = streamlit.selectbox("RainToday", df.RainToday.unique())
    MinTemp = streamlit.number_input("MinTemp")
    Rainfall = streamlit.number_input("Rainfall")
    WindGustSpeed = streamlit.number_input("WindGustSpeed")
    WindSpeed3pm = streamlit.number_input("WindSpeed3pm")
    Humidity9am = streamlit.number_input("Humidity9am")
    Humidity3pm = streamlit.number_input("Humidity3pm")
    Pressure9am = streamlit.number_input("Pressure9am")
    Date = streamlit.text_input("Ingresa la fecha de interés.", "Ejemplo: 2008-12-31")

    data = {
            "Location": Location,
            "WindGustDir": WindGustDir,
            "WindDir9am": WindDir9am,
            "WindDir3pm": WindDir3pm,
            "RainToday": RainToday,
            "MinTemp": MinTemp,
            "Rainfall": Rainfall,
            "WindGustSpeed": WindGustSpeed,
            "WindSpeed3pm": WindSpeed3pm,
            "Humidity9am": Humidity9am,
            "Humidity3pm": Humidity3pm,
            "Pressure9am": Pressure9am,
            "Date": Date
            }

    if streamlit.button("Predict"):
        response = requests.post("http://0.0.0.0:8000/predict", json=data)
        prediction = response.text
        streamlit.success(f"The prediction from model: {prediction}")


if __name__ == '__main__':
    # by default it will run at 8501 port
    run()
