import argparse
import os
import os.path as op

from encoder.dataset.datasets import *
from heads import *
from encoder.dataset.tools.config import load_config
from training.trainer import Trainer
from training.init_tools import prepare_out_folder, load_train_objs, prepare_dataloader, load_eval_objs



def none_or_str(value):
    if value == 'None':
        return None
    return value
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate motion head")
    parser.add_argument("config", type=str, default="encoder/dataset/tools/default_config.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to the checkpoint where resuming")
    parser.add_argument('--frozen_encoder', type=none_or_str,  default=None, help="Whether to import the weights of the encoder and freeze its weights")
    parser.add_argument('--params', type=str, default=None, help='set if some parameters should be loaded from yaml')
    parser.add_argument('--force', action="store_true", default=False, help="Whether to delete existing checkpoints if exist yet")
    parser.add_argument('--no_eval', action="store_true", default=False, help="Whether to skip evaluation or not")
    parser.add_argument('--output_folder', type=str, default=None, help="Path where saving checkpoints")
    parser.add_argument('--unfreeze', action="store_true", default=False, help="Unfreeze the frozen encoder, specific to pretrained weights")
    parser.add_argument('--trainset_name', type=str, default="trainset", help="Name of the training set to be used for training")
    parser.add_argument('--testset_name', type=str, default="testset", help="Name of the training set to be used for testing")
    parser.add_argument('--gt_name', type=str, default='label', help="Name of the gt variable to compare the output with")
    parser.add_argument('--load_frozen_head', action="store_true", default=False, help="Load head part of frozen encoder")
    parser.add_argument('--gpu_id_eval', type=int, default=0, help="Id of the gpu where running evaluation (and trining in case of single gpu)")
    parser.add_argument('--force_classif', action="store_true", default=False, help="force the result to be done on classif output if double output")
    parser.add_argument('--dvc_metrics', action="store_true", default=False, help="Save metrics in dvc")
    return parser.parse_args()


def main_single_gpu(config, args):
    device = torch.device(args.gpu_id_eval)
    encoder, head, dataset, optimizer, scheduler, epoch_start, text_encoder = load_train_objs(config, device, args)
    dl = prepare_dataloader(dataset, config, "train")
    dl_val = None if args.no_eval else load_eval_objs(config, args)

    out_train_path, lambda_create_out_folder = prepare_out_folder(args)

    trainer = Trainer(encoder, head, dl, optimizer, scheduler, config, out_train_path, device,
                      epoch_start=epoch_start, dl_val=dl_val, gt_name=args.gt_name, gpu_id_eval=args.gpu_id_eval,
                      text_encoder=text_encoder, force_classif=args.force_classif)
    print("*****************")
    print("****** Train single gpu ******")
    trainer.train(lambda_create_out_folder, args.dvc_metrics)
    print("results in : ", out_train_path)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, update_default=False, params_path=args.params)
    assert "eval_interval" in config["train"]["settings"]

    main_single_gpu(config, args)
