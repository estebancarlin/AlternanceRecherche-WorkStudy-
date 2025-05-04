import argparse
import os
import yaml
from easydict import EasyDict
from src.train import train
from src.test import test
from src.prediction import prediction


def load_config(path='configs/config.yaml'):
    """Load configuration file as an EasyDict object."""
    with open(path, 'r') as stream:
        return EasyDict(yaml.safe_load(stream))


def run_train(config_path):
    """Run training with the specified configuration path."""
    config = load_config(config_path)
    train(config)


def run_test(path):
    """Run testing using the specified experiment path."""
    config = load_config(os.path.join(path, 'config.yaml'))
    test(path, config)


def run_prediction(experiment_path, irm_path):
    """Run prediction using weights and IRM path."""
    config = load_config(os.path.join(experiment_path, 'config.yaml'))
    weights_path = os.path.join(experiment_path, 'model15.pth')
    prediction(config, weights_path, irm_path)


def main(options):
    """Main execution logic based on the selected mode."""
    mode = options.get('mode')
    
    if mode == 'train':
        run_train(options['config_path'])
    elif mode == 'test':
        run_test(options['path'])
    elif mode == 'prediction':
        run_prediction(options['experiment_path'], options['path'])
    else:
        raise ValueError("Invalid mode. Choose from: 'train', 'test', or 'prediction'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training, testing, or prediction pipeline.")
    
    parser.add_argument('--mode', required=True, choices=['train', 'test', 'prediction'], help="Execution mode.")
    parser.add_argument('--config_path', default='configs/config.yaml', type=str, help="Path to the config file.")
    parser.add_argument('--experiment_path', type=str, help="Path to the experiment folder (for prediction).")
    parser.add_argument('--path', type=str, help="Path for test or IRM path for prediction.")

    args = parser.parse_args()
    main(vars(args))
