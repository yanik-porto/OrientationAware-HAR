import yaml
import os.path as op

def load_config_yaml(path):
    with open(path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    return config


# Recursive function to update the configuration
def update_config(default, new_config):
    for key, value in new_config.items():
        if isinstance(value, dict) and key in default:
            update_config(default[key], value)
        else:
            default[key] = value

def load_config(path, params_path=None, update_default=True):
    config_to_load = load_config_yaml(path)

    # add loaded config to default one
    if update_default:
        config_default = load_config_yaml(op.join(op.dirname(__file__), "default_config.yaml"))
        update_config(config_default, config_to_load)
        config_to_load = config_default

    if params_path is not None:
        params = yaml.safe_load(open(params_path))
        update_with_params(config_to_load, params)

    return config_to_load

def update_subconfig(config, params, subname):
    if subname in params:
        if 'config' in params[subname]:
            encoder_config = load_config_yaml(params[subname]['config'])
        else:
            encoder_config = params[subname]
        config[subname] = encoder_config

def update_with_params(config, params):
    if 'encoder' in params:
        update_subconfig(config, params, 'encoder')

    if 'head' in params:
        update_subconfig(config, params, 'head')

    if 'train' in params:
        update_config(config['train'], params['train'])
    
def autotune_config(config, setname):
    if 'n_views' in config:
        if setname in config:
            config[setname]['dataset']['params']['n_views'] = config['n_views']

    if setname in config:
        n_views = config[setname]['dataset']['params']['n_views']

        n_persons = 1
        if 'FormatGCNInput' in config[setname]['preprocessing']:
            if 'num_person' in  config[setname]['preprocessing']['FormatGCNInput']:
                n_persons = config[setname]['preprocessing']['FormatGCNInput']['num_person']

        if 'num_person' in config['encoder']['params']:
            config['encoder']['params']['num_person'] = n_persons * n_views

        if 'n_persons' in config['head']['params']:
            config['head']['params']['n_persons'] = n_persons

        if 'n_views' in config['head']['params']:
            config['head']['params']['n_views'] = n_views

    return config
