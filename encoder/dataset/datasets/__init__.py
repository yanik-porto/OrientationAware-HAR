from .dataset_mv import DatasetMV
from .dataset_babel_mv import DatasetBabelMV
from .dataset_mv_negative import DatasetMVNegative

from preprocessors import *

import os.path as op

def create_dataset(config, setname="train12"):
    assert setname in config, "no " + setname + " specified in config"
    assert 'dataset' in config[setname], "no dataset specified in config"

    preprocessing = create_preprocessing(config[setname])

    cfg_dataset = config[setname]['dataset']

    name = cfg_dataset.get('name', None)
    use_negatives = cfg_dataset.get('use_negatives', False)

    dataset = None
    if name in ("babel_mv", "babel", "ntu","ts","nucla","rhm_har","mcad","posetics","tsu_pretrain","pkummd"):
        if use_negatives:
            dataset = DatasetMVNegative(op.join(op.dirname(__file__), cfg_dataset['file']), name, split=cfg_dataset["split"], preprocessing=preprocessing, **cfg_dataset["params"])
        else:
            dataset = DatasetBabelMV(op.join(op.dirname(__file__), cfg_dataset['file']), split=cfg_dataset["split"], preprocessing=preprocessing, **cfg_dataset["params"])
    else:
        print(name, " not handled yet in datasets")

    return dataset