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

# Runs adaptive version of Texar Bert model.

if ! python3 -m pip show adaptdl-cli > /dev/null 2>&1
then python3 -m pip install adaptdl-cli
fi

ROOT=$(dirname $0)/../..
cat << EOF | adaptdl submit $ROOT -d $ROOT/docker/Dockerfile -f -
apiVersion: adaptdl.petuum.com/v1
kind: AdaptDLJob
metadata:
  generateName: texar-bert-elastic-
spec:
  template:
    spec:
      containers:
      - name: main
        command:
        - python3
        - examples/bert/bert_classifier_adaptive.py
        - --do-train
        - --do-eval
        - --config-downstream=config_classifier
        - --config-data=config_data
        - --output-dir=output
        env:
        - name: PYTHONUNBUFFERED
          value: "true"
        resources:
          limits:
            nvidia.com/gpu: 1
EOF
