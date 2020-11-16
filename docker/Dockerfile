# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


FROM python:3.7-slim
WORKDIR /root

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

COPY . texar-pytorch
WORKDIR texar-pytorch

RUN python3 setup.py bdist_wheel
ARG TEXAR_VERSION=0.0.0
RUN TEXAR_VERSION=${TEXAR_VERSION} pip install dist/*.whl
RUN pip install -r requirements.txt

RUN pip install tensorflow adaptdl>=0.2.4 tensorboard
RUN rm -rf dist

ENV PYTHONUNBUFFERED=true
