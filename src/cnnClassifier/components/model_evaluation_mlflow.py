import os
import json
import tensorflow as tf
import mlflow
from datetime import datetime
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig


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

    def _build_model(self):
        """Rebuild VGG16 + custom head model architecture."""
        base = tf.keras.applications.VGG16(
            input_shape=(224, 224, 3),
            weights=None,
            include_top=False
        )
        x = tf.keras.layers.Flatten()(base.output)
        out = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs=base.input, outputs=out)
        return model

    def load_model(self, path):
        """Load model by rebuilding architecture + loading weights."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        file_size = os.path.getsize(path)
        logger.info(f"Model file size: {file_size:,} bytes")

        if file_size < 1000:
            raise ValueError(f"Model file too small ({file_size} bytes), likely empty/corrupt")

        model = self._build_model()

        try:
            model.load_weights(path)
            logger.info("Model weights loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            raise

        return model

    def evaluation(self):
        path = self.config.path_of_model
        logger.info(f"Loading model from: {path}")

        self.model = self.load_model(path)
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        logger.info("Model loaded and compiled!")
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        logger.info(f"Loss: {self.score[0]}, Accuracy: {self.score[1]}")

    def log_into_mlflow(self):
        try:
            # Set credentials FIRST
            os.environ["MLFLOW_TRACKING_USERNAME"] = "Anshika102005"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = "0ba6edfa35e8c01b7f7fa274d3bafcbfd35dac2c"

            # Set tracking URI directly (skip dagshub.init)
            tracking_uri = "https://dagshub.com/Anshika102005/Kidney-Disease-Classification-DL-Project-DVC.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow Tracking URI: {tracking_uri}")

            # Set experiment
            experiment_name = "Kidney-Disease-Classification"
            mlflow.set_experiment(experiment_name)
            logger.info(f"Experiment set: {experiment_name}")

            # Unique run name
            run_name = f"kidney-experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            logger.info(f"Starting MLflow run: {run_name}")

            with mlflow.start_run(run_name=run_name) as run:
                # Log all parameters
                mlflow.log_params(self.config.all_params)

                # Log metrics
                mlflow.log_metrics({
                    "loss": self.score[0],
                    "accuracy": self.score[1]
                })

                # Log model
                mlflow.keras.log_model(self.model, "model")

                logger.info(f"Run ID: {run.info.run_id}")
                logger.info(f"Run Name: {run_name}")
                logger.info(f"Experiment ID: {run.info.experiment_id}")

                # Save scores to JSON
                scores = {"loss": self.score[0], "accuracy": self.score[1]}
                scores_path = os.path.join(os.getcwd(), "scores.json")
                with open(scores_path, "w") as f:
                    json.dump(scores, f, indent=2)
                logger.info(f"Scores saved to {scores_path}")

                # Log JSON as artifact
                mlflow.log_artifact(scores_path)

                logger.info("✅ MLflow logging successful!")

        except Exception as e:
            logger.error(f"❌ MLflow logging failed: {e}")
            import traceback
            traceback.print_exc()
            raise
