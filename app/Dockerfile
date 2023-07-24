FROM python:3.8-slim-bullseye as build

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -U pip && pip install -r /app/requirements.txt 

EXPOSE 8000

COPY config.yaml main.py prediction_functions.py rain_model.sav /app/

CMD ["python", "main.py"]