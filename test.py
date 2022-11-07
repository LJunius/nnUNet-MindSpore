from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = '/home/ictpercomp/sdb1/chengs18/nnunet_dataset_ms/nnUNet_preprocessed/Task040_KiTS/nnUNetPlansv2.1_plans_3D.pkl'
    output_file = '/home/ictpercomp/sdb1/chengs18/nnunet_dataset_test/nnUNet_preprocessed/Task040_KiTS/nnUNetPlansv2.1_plans_3D.pkl'
    # output_file = '/home/ictpercomp/sdb1/chengs18/nnunet_dataset/nnUNet_preprocessed/Task040_KiTS/nnUNetPlansv2.1_plans_3D.pkl'
    a = load_pickle(input_file)
    b = load_pickle(output_file)
    b['plans_per_stage'] = a['plans_per_stage']
    a['plans_per_stage'][0]['batch_size'] = int(2)

    # a['plans_per_stage'][1]['batch_size'] = int(2)
    save_pickle(b, output_file)
    print("ok")