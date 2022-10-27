# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:39:23 2022

@author: ASUS
"""
import numpy as np
import argparse
import os
from batchgenerators.utilities.file_and_folder_operations import subfiles
import SimpleITK as sitk

def Dice(predict, label, eps=1e-6):
    inter = np.sum(predict * label)
    union = np.sum(predict) + np.sum(label)
    dice = (2*inter + eps) / (union + eps)
    return dice


def Hec_dice(predict, label):
    """
    0:其他组织 1:肾脏 2:肿瘤
    hec1:肾脏+肿瘤
    hec2:肿瘤
    """
    hec1 = Dice(predict > 0, label > 0)
    hec2 = Dice(predict == 2, label == 2)
    return (hec1 + hec2) / 2, hec1, hec2


def compute_dice(parser):
    args = parser.parse_args()
    input_folder = args.input_folder
    label_folder = args.label_folder
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    all_score = 0
    all_hec1 = 0
    all_hec2 = 0
    for file in all_files:
        file_path = os.path.join(input_folder, file)
        label_name = file.split(".")[0]
        label_path = os.path.join(label_folder, label_name+".nii.gz")
        predict = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        score, hec1, hec2 = Hec_dice(predict, label)
        all_score += score
        all_hec1 += hec1
        all_hec2 += hec2
        print(f"{label_name}, score:{score}, hec1:{hec1}, hec2:{hec2}")
    return all_score/len(all_files), all_hec1/len(all_files), all_hec2/len(all_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=False, default="/home/ictpercomp/sdb1/chengs18/nnunet_dataset/nnUNet_raw/nnUNet_val_data/predict_torch",
                        help="the result folder of predicted fiile like xx.nii.gz")
    parser.add_argument("--label_folder", type=str, required=False, default="/home/ictpercomp/sdb1/chengs18/nnunet_dataset/nnUNet_raw/nnUNet_raw_data/Task040_KiTS/labelsTr",
                        help="the label folder")

    ave_score, ave_hec1, ave_hec2 = compute_dice(parser)
    print(f'ave_score:{ave_score}, ave_hec1:{ave_hec1}, ave_hec2:{ave_hec2}')