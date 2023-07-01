import argparse
import json

from bin.coc_model import coc_Model
from bin.skin_model import skin_Model

def main(config_path):

    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise ValueError("Configuration file not found.")

    case = config['case']
    
    if case == 'skin':

        model = skin_Model(config)
        model.train_model()

    if case == 'coc':
        
        model = coc_Model(config)
        model.train_model()

    if case == 'coc3':
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Maximum Likelihood Ground Truth')
    parser.add_argument('config', type = str, help = 'Configuration file path.')
    args = parser.parse_args()

    main(args.config)