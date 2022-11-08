import shutil
from batchgenerators.utilities.file_and_folder_operations import *
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', type=str,
                        default="/home/ma-user/work/data/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task040_KiTS/test_predict",
                        help="Must contain all modalities for each patient in the correct", required=False)
    parser.add_argument("--zero_to_nozero", default=True, type=bool,
                        help="if True, rename case_000**_0000.nii.gz to case_000**.nii.gz"
                             "else rename case_000**.nii.gz to case_000**_0000.nii.gz")
    args = parser.parse_args()
    zero_to_nozero = args.zero_to_nozero
    base = args.input_folder
    if zero_to_nozero:
        cases = subfiles(base, join=False, suffix="nii.gz")
        for c in cases:
            nc = c.split('.')[0]
            if nc.endswith("_0000") is False:
                print(f"{c} not endwith _0000")
                break
            nc = nc.split('_')[0] + '_' + nc.split('_')[1] + '.nii.gz'
            src = os.path.join(base, c)
            dst = os.path.join(base, nc)
            shutil.move(src, dst)
            print(src,"....",dst)
    else:
        cases = subfiles(base, join=False, suffix="nii.gz")
        for c in cases:
            nc = c.split('.')[0]
            if nc.endswith("_0000") is True:
                print("already endwith _0000")
                break
            nc = nc + "_0000.nii.gz"
            src = os.path.join(base, c)
            dst = os.path.join(base, nc)
            shutil.move(src, dst)
            print("before rename: ", src,"after rename: ",dst)
