from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity import DataIngestionConfig, PrepareBaseModelConfig  # Fixed import
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        artifacts_root = self.config.get("artifacts_root", self.config["data_ingestion"]["root_dir"])
        create_directories([artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config["root_dir"]),
            source_URL=config["source_URL"],
            local_data_file=Path(config["local_data_file"]),
            unzip_dir=Path(config["unzip_dir"]) 
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params.get("TrainingArguments", self.params.get("training_arguments", self.params))
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=params.get("image_size", params.get("IMAGE_SIZE")),
            params_learning_rate=params.get("learning_rate", params.get("LEARNING_RATE")),
            params_include_top=params.get("include_top", params.get("INCLUDE_TOP")),
            params_weights=params.get("weights", params.get("WEIGHTS")),
            params_classes=params.get("classes", params.get("CLASSES"))
        )

        return prepare_base_model_config
    