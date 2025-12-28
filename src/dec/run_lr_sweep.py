import argparse
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DEC learning rate sweep experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("Loaded config successfully.")
    print(f"Experiment name: {config['experiment']['name']}")


if __name__ == "__main__":
    main()