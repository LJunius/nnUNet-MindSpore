#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

export DISTRIBUTE=0
export DEVICE_ID=2
export nnUNet_raw_data_base="/home/i/sdb1/chengs18/nnunet_dataset/nnUNet_raw/nnUNet_val_data/imagesTr"
export nnUNet_preprocessed="/home/ictpercomp/sdb1/chengs18/nnunet_dataset/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/ictpercomp/sdb1/chengs18/nnunet_dataset/nnUNet_trained_models"

python eval.py

