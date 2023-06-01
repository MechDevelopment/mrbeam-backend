FROM python:3.9-slim-buster

ENV PYTHONUNBUFFERED=1

RUN apt-get update

RUN mkdir -p /app

WORKDIR /app
COPY ./requirements.txt /app

RUN pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY ./app /app

COPY ./weights /weights

EXPOSE 8000
