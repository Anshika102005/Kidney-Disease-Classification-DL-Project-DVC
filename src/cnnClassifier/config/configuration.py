from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    # ------------------------------------------------------------------ #
    #  Data Ingestion
    # ------------------------------------------------------------------ #
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    # ------------------------------------------------------------------ #
    #  Prepare Base Model
    # ------------------------------------------------------------------ #
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params.TrainingArguments
        create_directories([config.root_dir])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=params.image_size,
            params_learning_rate=params.learning_rate,
            params_include_top=params.include_top,
            params_weights=params.weights,
            params_classes=params.classes,
        )

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params.TrainingArguments
        prepare_base_model = self.config.prepare_base_model
        training_data = (
            Path(self.config.data_ingestion.unzip_dir) / "kidney-ct-scan-image"
        )
        create_directories([config.root_dir])
        return TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=training_data,
            params_epochs=params.epochs,
            params_batch_size=params.batch_size,
            params_is_augmentation=params.augmentation,
            params_image_size=params.image_size,
        )

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    def get_evaluation_config(self) -> EvaluationConfig:
        params = self.params.TrainingArguments
        return EvaluationConfig(
            path_of_model=Path("artifacts/training/model.h5"),
            training_data=Path("artifacts/data_ingestion/kidney-ct-scan-image"),
            all_params=self.params,
            mlflow_uri="https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow",
            params_image_size=params.image_size,
            params_batch_size=params.batch_size,
        )
