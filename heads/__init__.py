from .head_text_embed import *
from .head_base import *
from .head_pool import *

def create_head(config):
    assert 'head' in config, "no head specified in config"

    cfg_head = config['head']

    name = cfg_head.get('name', None)

    head = None
    if name == "base":
        head = HeadBase(**cfg_head["params"])
    elif name == "pool":
        head = HeadPool(**cfg_head["params"])
    elif name == "textembed":
        head = HeadTextEmbed(**cfg_head["params"])
    elif name == "textembeddualoutput":
        head = HeadTextEmbedDualOutput(**cfg_head["params"])
    else:
        print(name, " not handled yet in heads")

    return head