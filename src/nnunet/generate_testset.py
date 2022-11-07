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

"""generate test set for validate model"""

import os
import argparse
import shutil
from batchgenerators.utilities.file_and_folder_operations import join, \
    isfile, save_pickle, load_pickle, maybe_mkdir_p, \
    os
import numpy as np



def do_generate_testset(par):
    """generate testset"""
    args = par.parse_args()
    split_file = np.load(args.splits_final, allow_pickle=True)
    val_list = split_file[args.fold]["val"]
    output_dir = "/home/ictpercomp/sdb1/chengs18/nnunet_dataset_test/test_dataset"
    raw_data_path = "/home/ictpercomp/sdb1/chengs18/nnunet_dataset_torch/nnUNet_raw/nnUNet_raw_data/Task040_KiTS/imagesTr"
    for case in val_list:
        src = join(raw_data_path, case + '_0000.nii.gz')
        dst = join(output_dir, case+'.nii.gz')
        print(src, "...", dst)
        shutil.copy(src, dst)
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_final", type=str,
                        default="/home/ictpercomp/sdb1/chengs18/nnunet_dataset_torch/nnUNet_preprocessed/Task040_KiTS/splits_final.pkl",
                        required=False, help="split file path")
    parser.add_argument("--fold", type=int, default=3, required=False, help="which fold for validate")
    do_generate_testset(par=parser)
