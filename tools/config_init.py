from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args) #0.7second save
    return cfg

import pickle
#only save config
config_dir = './server_config.pkl'
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    with open(config_dir, 'wb') as file:
        pickle.dump(cfg, file)

