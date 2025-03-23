import os
import yaml
from dotenv import load_dotenv

def load_env(config_path: str = "config.yml"):


    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            try:
                config = yaml.safe_load(file) or {}
                for key, value in config.items():
                    if key.startswith("#") or value is None:
                        continue
                    os.environ[key] = str(value)
                print(f"Loaded environment variables from {config_path}")
            except yaml.YAMLError as e:
                print(f"Failed to load config.yaml: {e}")
    else:
        print(f"Config file not found at {config_path}")
