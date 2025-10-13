FROM python:3.12-slim

RUN apt-get update -y

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install "fastapi[standard]"
RUN pip install requests

COPY main.py .
