FROM python:3.9-slim-buster

ENV SHELL /bin/bash

WORKDIR LPR

RUN apt-get update

COPY requirements.txt WORKDIR/requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install pip==21.3.1
RUN pip install -r WORKDIR/requirements.txt

RUN pip install numpy --upgrade
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
CMD ["tail", "-f", "/dev/null"]