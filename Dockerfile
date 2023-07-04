FROM ubuntu:latest

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends\
 && apt-get install -y --no-install-recommends\
    python3 python3-venv wget python3-opencv\
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    qt5-gtk-platformtheme

COPY main.py .
COPY video.mp4 .
COPY requirements.txt .
COPY pose_landmarker_lite.task .
COPY model_detection.py .
COPY plane_detection.py .
COPY intersection_result.py .

RUN apt-get install -y python3-pip

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/requirements.txt

ARG USERNAME=mediapipe
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
    
CMD ["python3", "main.py", "-v", "video.mp4"]
