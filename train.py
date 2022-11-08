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

"""train module"""

import argparse

from batchgenerators.utilities.file_and_folder_operations import os
from mindspore import context
from mindspore.communication import init
from mindspore.context import ParallelMode

from src.nnunet.paths import default_plans_identifier
from src.nnunet.run.default_configuration import get_default_configuration
from src.nnunet.run.load_pretrained_weights import load_pretrained_weights
from src.nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from src.nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

import wandb

# os.environ['DEVICE_ID'] = '1'
# os.environ['RANK_SIZE'] = '1'
os.environ['DISTRIBUTE'] = '0'
def do_train(parser):
    wandb.init(project="HW-RESULT", entity="dog-left")
    # """train logic according to parser args"""
    args = parser.parse_args()
    task = args.task
    fold = args.fold

    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_distribute = int(os.getenv('DISTRIBUTE'))
    if run_distribute == 1:
        context.set_context(device_id=device_id)  # set device_id
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True, gradients_mean=True)

        init()

    else:

        context.set_context(device_id=device_id)  # set device_id

    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if trainer_class is not None:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)

    if args.disable_saving:
        trainer.save_final_checkpoint = False
        trainer.save_best_checkpoint = False
        trainer.save_intermediate_checkpoints = True
        trainer.save_latest_only = True

    trainer.initialize(not validation_only)

    config = {
        "learning_rate": trainer.lr_scheduler_eps,
        "epochs": trainer.max_num_epochs,
        "batch_size": trainer.batch_size,
        "fold": trainer.fold,
        "network": args.network,
        "trainer": args.network_trainer,
        "fp32": args.fp32,
    }
    wandb.config.update(config)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                trainer.load_latest_checkpoint()
            elif (not args.continue_training) and (args.pretrained_weights is not None):
                load_pretrained_weights(trainer.network, args.pretrained_weights)
            else:
                pass
            trainer.run_training()

        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                # trainer.load_final_checkpoint(train=False)
                filename = os.path.join(args.model_folder_name, f'fold_{args.fold}', "model_final_checkpoint.ckpt")
                if not os.path.isfile(filename):
                    raise RuntimeError(
                        "Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
                trainer.load_checkpoint(filename, train=False)

    trainer.validate(save_softmax=args.npz, validation_folder_name=args.val_folder,
                     run_postprocessing_on_folds=not args.disable_postprocessing_on_folds,
                     overwrite=args.val_disable_overwrite, do_infer=True, predict_output_folder=args.predict_output_folder)

def main():
    """train logic"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-network", default="3d_fullres", required=False)
    parser.add_argument("-network_trainer", default="nnUNetTrainerV2", required=False)
    parser.add_argument("-task", default="Task040_KiTS", help="can be task name or task id", required=False)
    parser.add_argument("-fold", default=1, help='0, 1, ..., 5 or \'all\'', required=False)
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("--predict_output_folder", help="run second predict, the output dir",
                        default=None, required=False)
    parser.add_argument("--model_folder_name", type=str, required=False,
                        default="/home/ictpercomp/sdb1/chengs18/nnunet_dataset_ms/nnUNet_trained_models/nnUNet/3d_fullres/Task040_KiTS/nnUNetTrainerV2__nnUNetPlansv2.1",
                        help="precise model name")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, "
                             "the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw_test2",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model).'
                             'Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')


    do_train(parser)

if __name__ == "__main__":
    main()
