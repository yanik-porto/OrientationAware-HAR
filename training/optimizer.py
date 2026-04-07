import torch.optim as optim

def create_optimizer(config, encoder, head, text_encoder=None):
    assert 'optimizer' in config["train"], "no optimizer specified in config"

    cfg_optim = config["train"]['optimizer']

    name = cfg_optim.get('name', None)

    params = [{"params": filter(lambda p: p.requires_grad, encoder.parameters()), "lr": cfg_optim["lr"]},
              {"params": filter(lambda p: p.requires_grad, head.parameters()), "lr": cfg_optim["lr_head"] if "lr_head" in cfg_optim else cfg_optim["lr"]}]

    if text_encoder is not None:
        params.append({"params": filter(lambda p: p.requires_grad, text_encoder.parameters()), "lr": cfg_optim["lr"]})

    optimizer = None
    if name == "sgd":
        optimizer = optim.SGD(params,  lr=cfg_optim["lr"], momentum=0.9, weight_decay=0.0005)
    elif name == "adam":
        optimizer = optim.Adam(params, lr=cfg_optim["lr"])
    elif name == "adamw":
        optimizer = optim.AdamW(params, lr=cfg_optim["lr"], weight_decay=0.01)
    else:
        print(name, " not handled yet in optimizers")

    return optimizer