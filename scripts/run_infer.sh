#!/bin/bash
# bash run_infer.sh /home/ictpercomp/sdb1/chengs18/nnunet_dataset_test/test_dataset /home/ictpercomp/sdb1/chengs18/nnunet_dataset_test/test_dataset/output_dir
database=$1
echo $1
echo $2

imagesTr_folder="$database/nnUNet_raw_data/Task040_KiTS"
presudo_labels_dir="$database/nnUNet_raw_data/Task040_KiTS/labelsTr"
raw_predict_dir="$database/nnUNet_raw_data/"
cropped_predict_dir="$database/nnUNet_cropped_data/"
preprocessing_predict_dir="$database/nnUNet_preprocessed/"
first_predict_model="/home/ictpercomp/sdb1/chengs18/nnunet_dataset_ms/nnUNet_trained_models/nnUNet/3d_fullres/Task040_KiTS/nnUNetTrainerV2__nnUNetPlansv2.1"
second_predict_model="/home/ictpercomp/sdb1/chengs18/nnunet_dataset_ms/nnUNet_trained_models/nnUNet/3d_fullres/Task040_KiTS/nnUNetTrainerV2__nnUNetPlansv2.1"
echo $imagesTr_folder
echo $presudo_labels_dir

mkdir -p $2
mkdir -p $imagesTr_folder
mkdir -p $presudo_labels_dir
mkdir -p $raw_predict_dir
mkdir -p $cropped_predict_dir
mkdir -p preprocessing_predict_dir

#python eval.py --input_folder $1 --output_folder $presudo_labels_dir --model_folder_name $first_predict_model --folds 3
#python src/nnunet/dataset_conversion/Task040_KiTS_new.py --input_folder $1 --output_folder $imagesTr_folder
#python src/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py --raw_predict_dir $raw_predict_dir --cropped_predict_dir $cropped_predict_dir --preprocessing_predict_dir $preprocessing_predict_dir
python train.py --predict_output_folder $2 --model_folder_name $second_predict_model
