import os
import os.path as op
from os.path import dirname as opd
from os.path import basename as opb
import shutil
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from encoder.encoders import create_encoder
from encoder.dataset.datasets import *
from heads import *
from tools.checkpoint import adapt_batch_norm, num_params, clean_checkpoint
from training.optimizer import create_optimizer
from torch.utils.data.distributed import DistributedSampler

def prepare_out_folder(args):
    foldername = op.splitext(op.basename(args.config))[0]
    if opb(opd(opd(args.config))) == 'configs':
        foldername = op.join(opb(opd(args.config)), foldername)
    out_train_path = op.join("output", foldername + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
    print("output path : ", out_train_path)
    if not args.resume and args.output_folder is not None and op.islink(args.output_folder):
        if args.force:
            os.unlink(args.output_folder)
        else:
            assert len([name for name in os.listdir(out_train_path) if op.isfile(op.join(out_train_path, name))]) < 2, "Training was previously run"

    create_out_folder_later = lambda: create_out_folder(args, out_train_path) 
    return out_train_path, create_out_folder_later

def create_out_folder(args, out_train_path):
    if not os.path.exists(out_train_path):
        os.makedirs(out_train_path, exist_ok=True)

        # create symlink for output_folder
        if args.output_folder is not None:
            os.symlink(op.basename(out_train_path), args.output_folder)

        # TODO : copy config from memory. The config can change between time of loading and time of copying.
        # Moreover, it will lighten the copy with only used sets. 
        shutil.copy(args.config, op.join(out_train_path, op.basename(args.config)))

def load_train_objs(config, device, args):
    # load model
    print("* create model ... ", end="")
    encoder = create_encoder(config)
    head = create_head(config)
    encoder, head = encoder.to(device), head.to(device)
    print(f"# parameters in encoder : {num_params(encoder)}")
    print(f"# parameters in head : {num_params(head)}")

    text_encoder = None
    if 'encoder_text' in config:
        text_encoder = create_encoder(config, nodename='encoder_text')
        text_encoder.to(device)
        print(f"# parameters in text encoder : {num_params(text_encoder.model)}")

    if args.frozen_encoder is not None and not args.unfreeze:
        for p in encoder.parameters():
            p.requires_grad = False
        if text_encoder is not None:
            for p in text_encoder.model.parameters():
                p.requires_grad = False # Note: possibly overwritten by model.train()
    print("done")

    # create trainer
    print("* create trainer ...", end="")
    epoch_start = 0
    resume_chkpt = None
    if args.resume is not None:
        resume_chkpt = torch.load(args.resume)
        resume_chkpt['encoder'] = clean_checkpoint(resume_chkpt['encoder'], remove_fc=True)
        encoder.load_state_dict(resume_chkpt['encoder'], strict=True)
        head.load_state_dict(resume_chkpt['head'], strict=True)
        if 'text_encoder' in resume_chkpt:
            text_encoder.model.load_state_dict(resume_chkpt['text_encoder'], strict=True)
    elif args.frozen_encoder is not None:
        resume_chkpt = torch.load(args.frozen_encoder)
        adapt_batch_norm(resume_chkpt, encoder)
        resume_chkpt['encoder'] = clean_checkpoint(resume_chkpt['encoder'], remove_fc=True)
        encoder.load_state_dict(resume_chkpt['encoder'], strict=True)
        if 'text_encoder' in resume_chkpt and text_encoder is not None:
            text_encoder.model.load_state_dict(resume_chkpt['text_encoder'], strict=True)
        elif 'text_encoder' in resume_chkpt and text_encoder is None:
            print("Watchout, chkpt contains text_encoder but no text encoder configured")
        if args.load_frozen_head:
            head.load_state_dict(resume_chkpt['head'], strict=False)



    optimizer = create_optimizer(config, encoder, head, text_encoder)
    if args.resume is not None and resume_chkpt is not None:
        optimizer.load_state_dict(resume_chkpt['optimizer'])
        epoch_start = resume_chkpt["epoch"] + 1

    scheduler = MultiStepLR(optimizer, milestones=config["train"]['optimizer']["milestones"], gamma=config["train"]['optimizer']["lr_decay"], last_epoch=epoch_start-1)
    print("done")

    # load training dataset
    print("* create dataset ...", end="")
    dataset = create_dataset(config, setname=args.trainset_name)
    print(" done : ", len(dataset), " inputs")

    return encoder, head, dataset, optimizer, scheduler, epoch_start, text_encoder

def prepare_dataloader(dataset, config, mode="train", distribute=False):
    dl_settings = config[mode]["dataloader"]
    sampler = DistributedSampler(dataset, shuffle=True) if distribute else None
    shuffle = None if distribute else True
    return DataLoader(
        dataset,
        shuffle=shuffle,
        sampler=sampler,
        **dl_settings
    )

def load_eval_objs(config, args, distribute=False):
    # load eval dataset
    print("* create evaluation dataset ...", end="")
    dataset_val = create_dataset(config, setname=args.testset_name)
    print(" done : ", len(dataset_val), " inputs")

    dl = prepare_dataloader(dataset_val, config, "test", distribute=distribute)
    return dl