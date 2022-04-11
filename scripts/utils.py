import yaml
from easydict import EasyDict as edict


def load_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        res = yaml.safe_load(f)
    return edict(res)
