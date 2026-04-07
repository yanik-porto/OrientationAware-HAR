import os
import os.path as op
import torch

def clean_checkpoint(checkpoint, remove_fc=False):
    chkpt_clean = {}
    for key in checkpoint.keys():
        newkey = key.replace("module.", "")

        if remove_fc and 'fc.' in key:
        # if remove_fc and key.startswith('head.'):
            continue
        chkpt_clean[newkey] = checkpoint[key]
    return chkpt_clean

def create_symlink(path, name):
    symlink_path = op.join(op.dirname(path), name)
    if op.islink(symlink_path):
        os.unlink(symlink_path)
    os.symlink(op.basename(path), symlink_path)

def adapt_batch_norm(checkpoint, encoder):
    for key in checkpoint['encoder']:
        if "data_bn" in key and hasattr(encoder, "data_bn"):
            value = checkpoint['encoder'][key]
            if value.dim() == 0:
                continue

            chkpt_size = len(value)
            keyattr = key.replace("data_bn.", "")
            if hasattr(encoder.data_bn, keyattr):
                model_size = len(getattr(encoder.data_bn, keyattr))

                if model_size != chkpt_size and model_size % chkpt_size == 0:
                    values = (value,) * int(model_size / chkpt_size)
                    checkpoint['encoder'][key] = torch.cat(values)

def num_params(model):
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    return model_params

def greedy_soup(checkpoints):
    NUM_MODELS = len(checkpoints)
    for j, checkpoint in enumerate(checkpoints):
        if j == 0:
            uniform_soup = {k : v * (1./NUM_MODELS) for k, v in checkpoint.items()}
        else:
            uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in checkpoint.items()}
    return uniform_soup