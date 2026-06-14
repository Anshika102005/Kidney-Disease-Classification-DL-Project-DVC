import os
import json
import tensorflow as tf
import mlflow
import dagshub
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.load_trained_model import LoadTrainedModel


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def evaluation(self):
        self.model = LoadTrainedModel.load(model_path=self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        logger.info(f"Loss: {self.score[0]}, Accuracy: {self.score[1]}")

    def log_into_mlflow(self):
        dagshub.init(
            repo_owner="Anshika102005",
            repo_name="Kidney-Disease-Classification-DL-Project-DVC",
            mlflow=True
        )
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        with mlflow.start_run() as run:
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            logger.info(f"Run ID: {run.info.run_id}")
            scores = {"loss": self.score[0], "accuracy": self.score[1]}
            scores_path = os.path.join(os.getcwd(), "scores.json")
            with open(scores_path, "w") as f:
                json.dump(scores, f, indent=2)
