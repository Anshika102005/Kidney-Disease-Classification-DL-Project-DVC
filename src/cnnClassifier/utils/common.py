import os
import yaml
from cnnClassifier import logger
import json
import joblib
from pathlib import Path
from typing import Any
import base64
from box import Box


def read_yaml(path_to_yaml: Path) -> Box:
    """reads yaml file and returns a Box object for dot notation access"""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file) or {}
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return Box(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {path_to_yaml}\n{e}")
    except Exception as e:
        raise e
    

def create_directories(path_to_directories: list, verbose=True):
    """create list of directories"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def get_size(path: Path) -> str:
    """get size in KB"""
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"