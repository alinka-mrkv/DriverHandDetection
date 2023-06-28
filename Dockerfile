# a. Основа Ubuntu
FROM ubuntu:latest

WORKDIR /app

# Установка переменных окружения для автоматической установки часового пояса
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

# c. Установка python3, python3-venv, wget, python3-opencv  и необходимых библиотек через apt-get
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y python3 python3-venv wget python3-opencv 

# e. Создание нового пользователя
RUN useradd -ms /bin/bash newuser
USER newuser
WORKDIR /home/newuser

# Создание виртуального окружения и активация его
RUN python3 -m venv venv
RUN . venv/bin/activate

# Установка pip в виртуальное окружение
RUN venv/bin/pip install --upgrade pip

# b. Установка mediapipe
RUN venv/bin/pip install mediapipe==0.10.0

# Установка opencv-python через pip
# RUN venv/bin/pip install opencv-python

# d. Скачивание файла pose_landmarker.task
RUN wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Копирование файла main.py в Docker-образ
# COPY main.py .

# Запуск скрипта при старте контейнера
# CMD ["venv/bin/python3", "./main.py"]
