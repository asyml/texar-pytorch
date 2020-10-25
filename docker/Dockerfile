FROM python:3.7-slim
WORKDIR /root

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

COPY . texar-pytorch
WORKDIR texar-pytorch

RUN python3 setup.py bdist_wheel
ARG TEXAR_VERSION=0.0.0
RUN TEXAR_VERSION=${TEXAR_VERSION} pip install dist/*.whl

RUN pip install -r examples/bert/requirements.txt
RUN rm -rf dist

ENV PYTHONUNBUFFERED=true
