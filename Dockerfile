FROM python:2.7-slim

RUN apt-get update \
    && apt-get install -y gcc tk git

# COPY . /opt/app
# WORKDIR /opt/app

COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt
