FROM python:3.11-slim-buster

ENV PYTHONUNBUFFERED=1

RUN apt-get update
RUN apt-get install -y gcc libffi-dev libgl1 libglib2.0-0

RUN mkdir -p /app

WORKDIR /app
COPY ./requirements.txt /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./app /app

WORKDIR /

EXPOSE 8011
