from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = '/home/ma-user/work/data/nnunet_dataset/nnUNet_preprocessed/Task040_KiTS/nnUNetPlansv2.1_plans_3D.pkl'
    output_file = '/home/ma-user/work/data/nnunet_dataset/nnUNet_preprocessed/Task040_KiTS/nnUNetPlansv2.1_plans_3D_b4.pkl'
    a = load_pickle(input_file)
    a['plans_per_stage'][0]['batch_size'] = int(4)
    a['plans_per_stage'][1]['batch_size'] = int(4)
    save_pickle(a, output_file)