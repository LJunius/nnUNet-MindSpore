import shutil
from batchgenerators.utilities.file_and_folder_operations import *
import os
import argparse


def txt_read_line(file_name):
    """ 按行读取txt文件全部内容
    """
    with open(file_name) as fp:
        data_list = [_.strip() for _ in fp.readlines()]
    return data_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', type=str,
                        default="/home/ictpercomp/sdb1/chengs18/nnunet_dataset_test/test_dataset",
                        help="Must contain all modalities for each patient in the correct", required=False)
    parser.add_argument("--zero_to_nozero", default=True, type=bool,
                        help="if True, rename case_000**_0000.nii.gz to case_000**.nii.gz"
                             "else rename case_000**.nii.gz to case_000**_0000.nii.gz")
    parser.add_argument("--rename_type", default="output", type=str, help="value is 'input', 'output', 'mid'")
    parser.add_argument("--rename_file_dir", default="/home/ictpercomp/sdb1/chengs18/nnunet_dataset_test/test_dataset/output_dir", type=str, help="rename_file_dir")
    args = parser.parse_args()
    zero_to_nozero = args.zero_to_nozero
    base = args.input_folder
    rename_type = args.rename_type
    rename_file_dir = args.rename_file_dir
    if rename_type == "mid":
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
                print(src, "....", dst)
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
                print("before rename: ", src, "after rename: ", dst)
    elif rename_type == "input":
        cases = subfiles(base, join=False, suffix="nii.gz")
        f = open(join(rename_file_dir, "rename.txt"), 'a')
        for i, c in enumerate(cases):
            filename = 'case_{:0>5d}_0000.nii.gz'.format(i)
            src = os.path.join(base, c)
            dst = os.path.join(base, filename)
            f.write(c)
            f.write('\n')
            shutil.move(src, dst)
            print("before rename: ", src, "after rename: ", dst)
        f.close()
    elif rename_type == "output":
        cases = subfiles(base, join=False, suffix="nii.gz")
        cvts = txt_read_line(join(rename_file_dir, "rename.txt"))
        for i, c in enumerate(cases):
            src = os.path.join(base, c)
            filename = "Segmentation_" + cvts[i].split('.')[0] + '.nii.gz'
            dst = os.path.join(base, filename)
            shutil.move(src, dst)
            print("before rename: ", src, "after rename: ", dst)