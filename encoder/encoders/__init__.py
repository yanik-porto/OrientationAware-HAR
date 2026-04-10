from .aagcn import *
from .unik import *
from .protogcn import *
from .angle_conditioning import *
from .text_encoder import *

def create_encoder(config, nodename='encoder'):
    assert nodename in config, "no encoder specified in config"

    cfg_encoder = config[nodename]

    name = cfg_encoder.get('name', None)

    encoder = None
    if name == "aagcn":
        encoder = AAGCN(**cfg_encoder["params"])
    elif name == "unik":
        encoder = UNIK(**cfg_encoder["params"])
    elif name == "protogcn":
        encoder = ProtoGCN(**cfg_encoder["params"])
    elif name == "dualcond_aagcn":
        encoder = DualAngleConditioningAAGCN(**cfg_encoder["params"])
    elif name == "dualcond_unik":
        encoder = DualAngleConditioningUnik(**cfg_encoder["params"])
    elif name == "dualcond_protogcn":
        encoder = DualAngleConditioningProtogcn(**cfg_encoder["params"])
    elif name == "text_encoder":
        encoder = TextEncoder(**cfg_encoder["params"])
    else:
        print(name, " not handled yet in encoders")

    return encoder