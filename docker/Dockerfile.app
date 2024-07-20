FROM python:3.9-slim-buster

ENV SHELL /bin/bash
WORKDIR /LPR

# Update package list and install dependencies
RUN apt-get update && \
    apt-get install -y python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the application code and requirements
COPY requirements.txt requirements.txt
COPY . .

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install pip && \
    pip install -r requirements.txt && \
    pip install numpy --upgrade && \
    pip install opencv-python

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="${PYTHONPATH}:/LPR"

# Set the entry point for the container
ENTRYPOINT ["python3", "LPR/app.py"]
