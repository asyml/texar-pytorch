FROM python:3.7-slim
WORKDIR /root

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

COPY . texar-pytorch

WORKDIR texar-pytorch

RUN pip install -e .

RUN pip install -r examples/bert/requirements.txt

ENV PYTHONUNBUFFERED=true
