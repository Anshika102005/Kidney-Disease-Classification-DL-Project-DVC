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

        artifacts_root = self._resolve_path(
            self.config.get("artifacts_root", self.config["data_ingestion"]["root_dir"])
        )
        create_directories([artifacts_root])

    def _resolve_path(self, path_value) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return ROOT_DIR / path

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        root_dir = self._resolve_path(config["root_dir"])
        local_data_file = self._resolve_path(config["local_data_file"])
        unzip_dir = self._resolve_path(config["unzip_dir"])

        create_directories([root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=root_dir,
            source_URL=config["source_URL"],
            local_data_file=local_data_file,
            unzip_dir=unzip_dir
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params.get("TrainingArguments", self.params.get("training_arguments", self.params))
        root_dir = self._resolve_path(config.root_dir)
        
        create_directories([root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=root_dir,
            base_model_path=self._resolve_path(config.base_model_path),
            updated_base_model_path=self._resolve_path(config.updated_base_model_path),
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
        training_data = self._resolve_path(training.training_data)
        root_dir = self._resolve_path(training.root_dir)
        create_directories([
            root_dir
        ])

        training_config = TrainingConfig(
            root_dir=root_dir,
            trained_model_path=self._resolve_path(training.trained_model_path),
            updated_base_model_path=self._resolve_path(prepare_base_model.updated_base_model_path),
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
            path_of_model=self._resolve_path(self.config.training.trained_model_path),
            test_data=self._resolve_path(self.config.evaluation.test_data),
            all_params=dict(self.params),
            mlflow_uri=self.config.evaluation.mlflow_uri,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_classes=self.params.CLASSES
        )
        return eval_config
