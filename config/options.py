import json
from collections import OrderedDict

def parse(config_path):
    with open(config_path, 'r') as f:
        opt = json.load(f, object_pairs_hook=OrderedDict)

    return opt
