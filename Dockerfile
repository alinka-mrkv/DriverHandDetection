FROM ubuntu:latest

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends\
 && apt-get install -y --no-install-recommends\
 python3 python3-venv wget python3-opencv 

RUN apt-get install -y python3-pip
 
COPY requirements.txt .
COPY pose_landmarker_heavy.task .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/requirements.txt


ARG USERNAME=mediapipe
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
