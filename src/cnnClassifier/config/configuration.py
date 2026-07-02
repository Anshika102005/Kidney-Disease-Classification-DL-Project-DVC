from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig
from pathlib import Path
import os

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
            params_classes=params.get("classes", params.get("CLASSES")),
            params_freeze_all=params.get("freeze_all", params.get("FREEZE_ALL", False)),
            params_freeze_till=params.get("freeze_till", params.get("FREEZE_TILL", 0)),
            params_dropout=params.get("dropout", params.get("DROPOUT", 0.5))
        )

        return prepare_base_model_config
    


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = Path(training.training_data)
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=training_data,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_classes=params.CLASSES,
            params_early_stopping_patience=params.EARLY_STOPPING_PATIENCE,
            params_reduce_lr_patience=params.REDUCE_LR_PATIENCE,
            params_reduce_lr_factor=params.REDUCE_LR_FACTOR
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path(self.config.training.trained_model_path),
            training_data=Path(self.config.training.training_data),
            all_params=dict(self.params),
            mlflow_uri=self.config.evaluation.mlflow_uri,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_classes=self.params.CLASSES
        )
        return eval_config