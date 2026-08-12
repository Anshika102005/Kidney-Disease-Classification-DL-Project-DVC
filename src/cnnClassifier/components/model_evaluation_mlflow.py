import os
import io
import json
import re
import zipfile
from datetime import datetime
import tensorflow as tf
import mlflow
from tensorflow.keras.applications.vgg16 import preprocess_input
from cnnClassifier import config, logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _get_target_size(self):
        if hasattr(self, "model") and getattr(self.model, "input_shape", None):
            input_shape = self.model.input_shape
            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
                height, width = input_shape[1], input_shape[2]
                if isinstance(height, int) and isinstance(width, int):
                    return (height, width)

        image_size = self.config.params_image_size or []
        if len(image_size) >= 3:
            return tuple(int(dim) for dim in image_size[:2])
        return (224, 224)

    def _get_batch_size(self):
        batch_size = int(getattr(self.config, "params_batch_size", 16) or 16)
        return max(1, min(batch_size, 16))

    def _valid_generator(self):
        image_size_hw = self._get_target_size()
        batch_size = self._get_batch_size()

        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.test_data,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size_hw,
            batch_size=batch_size,
            shuffle=False,
        )

        def preprocess_test(images, labels):
            images = tf.cast(images, tf.float32)
            images = preprocess_input(images)
            return images, labels

        autotune = tf.data.AUTOTUNE
        # Cache speeds up repeated evaluation passes by avoiding repeated
        # decoding/resize work. Use in-memory cache (test set is small).
        self.valid_generator = (
            test_ds.map(preprocess_test, num_parallel_calls=autotune)
            .cache()
            .prefetch(autotune)
        )

    @staticmethod
    def _patch_input_layer():
        import keras.layers
        orig = keras.layers.InputLayer.__init__
        def _patched(self, input_shape=None, batch_shape=None, optional=False, **kwargs):
            if input_shape is None and batch_shape is not None:
                input_shape = batch_shape[1:] if batch_shape and len(batch_shape) > 1 else batch_shape
            orig(self, input_shape=input_shape, **kwargs)
        keras.layers.InputLayer.__init__ = _patched
        return orig

    @staticmethod
    def _restore_input_layer(orig):
        import keras.layers
        keras.layers.InputLayer.__init__ = orig

    @staticmethod
    def _patch_dtype_policy():
        import keras.mixed_precision.policy as policy_mod
        orig = policy_mod.deserialize
        def _patched(config, custom_objects=None):
            if isinstance(config, dict) and config.get("class_name") == "DTypePolicy":
                inner = config.get("config", {})
                return policy_mod.Policy(inner.get("name", "float32"))
            return orig(config, custom_objects)
        policy_mod.deserialize = _patched
        return orig

    @staticmethod
    def _restore_dtype_policy(orig):
        import keras.mixed_precision.policy as policy_mod
        policy_mod.deserialize = orig

    @staticmethod
    def _strip_numeric_suffix(name):
        return re.sub(r"_\d+$", "", name)

    @staticmethod
    def _h5_class_key(config_class_name):
        mapping = {
            "Conv2D": "conv2d",
            "DepthwiseConv2D": "depthwise_conv2d",
            "SeparableConv2D": "separable_conv2d",
            "Dense": "dense",
            "BatchNormalization": "batch_normalization",
            "LayerNormalization": "layer_normalization",
            "MaxPooling2D": "max_pooling2d",
            "AveragePooling2D": "average_pooling2d",
            "GlobalAveragePooling2D": "global_average_pooling2d",
            "GlobalMaxPooling2D": "global_max_pooling2d",
            "Dropout": "dropout",
            "InputLayer": "input_layer",
            "Flatten": "flatten",
            "Reshape": "reshape",
            "Concatenate": "concatenate",
            "Add": "add",
            "Multiply": "multiply",
            "Activation": "activation",
            "ReLU": "re_lu",
            "Softmax": "softmax",
            "Embedding": "embedding",
            "LSTM": "lstm",
            "GRU": "gru",
            "Bidirectional": "bidirectional",
            "TimeDistributed": "time_distributed",
        }
        return mapping.get(config_class_name, config_class_name.lower())

    @staticmethod
    def _natural_key(name):
        parts = re.split(r"(\d+)", name)
        result = []
        for p in parts:
            if p.isdigit():
                result.append((0, int(p)))
            else:
                result.append((1, p))
        return result

    @staticmethod
    def _get_weights_from_h5(raw_weights):
        import h5py

        raw_list = []
        buf = io.BytesIO(raw_weights)
        with h5py.File(buf, "r") as f:
            def collect(name, obj):
                if not isinstance(obj, h5py.Group):
                    return
                if "vars" not in obj:
                    return
                parts = name.split("/")
                if len(parts) == 2 and parts[0] == "layers":
                    grp_name = parts[1]
                    if grp_name == "optimizer":
                        return
                    num_vars = len(obj["vars"])
                    values = []
                    for i in range(num_vars):
                        values.append(obj["vars"][str(i)][()])
                    class_type = Evaluation._strip_numeric_suffix(grp_name)
                    raw_list.append((class_type, grp_name, values))
            f.visititems(collect)

        by_class = {}
        for class_type, grp_name, values in raw_list:
            by_class.setdefault(class_type, []).append((grp_name, values))
        for key in by_class:
            by_class[key].sort(key=lambda x: Evaluation._natural_key(x[0]))
            by_class[key] = [v for _, v in by_class[key]]

        return by_class

    @staticmethod
    def _convert_inbound_nodes(nodes):
        if not nodes or not isinstance(nodes, list):
            return nodes
        result = []
        for node in nodes:
            if isinstance(node, dict) and "args" in node:
                args = node.get("args", [])
                kwargs = node.get("kwargs", {})
                input_refs = []
                for arg in args:
                    if (
                        isinstance(arg, dict)
                        and arg.get("class_name") == "__keras_tensor__"
                    ):
                        history = arg.get("config", {}).get("keras_history", [])
                        if len(history) >= 3:
                            input_refs.append(
                                [history[0], history[1], history[2], kwargs]
                            )
                if input_refs:
                    result.append(input_refs)
            else:
                result.append(node)
        return result

    @staticmethod
    def _preprocess_config(config):
        for layer in config.get("config", {}).get("layers", []):
            if "config" in layer:
                cfg = layer["config"]
                bs = cfg.pop("batch_shape", None)
                if bs is not None:
                    if len(bs) > 0:
                        cfg["batch_size"] = bs[0]
                    if len(bs) > 1:
                        cfg["input_shape"] = bs[1:]
                cfg.pop("optional", None)
                cfg.pop("quantization_config", None)
                cfg.pop("sparse", None)
                cfg.pop("ragged", None)
                if isinstance(cfg.get("dtype"), dict):
                    cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")
            layer["inbound_nodes"] = Evaluation._convert_inbound_nodes(
                layer.get("inbound_nodes", [])
            )
        return config

    @staticmethod
    def _set_weights_from_h5(model, raw_weights):
        h5_weights = Evaluation._get_weights_from_h5(raw_weights)
        counters = {}

        for layer in model.layers:
            try:
                current = layer.get_weights()
            except Exception:
                current = None
            if not current or not layer.weights:
                continue

            class_type = Evaluation._h5_class_key(layer.__class__.__name__)
            idx = counters.get(class_type, 0)
            candidates = h5_weights.get(class_type, [])
            if idx < len(candidates):
                loaded = candidates[idx]
                if len(loaded) == len(current):
                    layer.set_weights(loaded)
                    counters[class_type] = idx + 1

    def evaluation(self):
        path = str(self.config.path_of_model)
        logger.info(f"Loading model from: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        else:
            self.model = tf.keras.models.load_model(path, compile=False)

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
        if not hasattr(self, "score"):
            self.evaluation()
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        scores_path = os.path.join(
            os.getcwd(),
            "scores.json"
            )
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
        logger.info(f"Scores saved to {scores_path}")
        try:
            import mlflow.keras

            tracking_uri = self.config.mlflow_uri
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow Tracking URI: {tracking_uri}")

            experiment_name = "Kidney-Disease-Classification"
            mlflow.set_experiment(experiment_name)
            logger.info(f"Experiment set: {experiment_name}")

            run_name = f"kidney-experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            logger.info(f"Starting MLflow run: {run_name}")

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics(scores)
                mlflow.keras.log_model(self.model, "model")

                logger.info(f"Run ID: {run.info.run_id}")
                logger.info(f"Run Name: {run_name}")
                logger.info(f"Experiment ID: {run.info.experiment_id}")

                mlflow.log_artifact(scores_path)
                logger.info("MLflow logging successful!")
        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("Continuing without MLflow logging.")


def main():
    # Set dummy credentials if not present (for local MLflow)
    # os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "local")
    # os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", "local")
    config = ConfigurationManager()
    eval_config = config.get_evaluation_config()
    evaluation = Evaluation(eval_config)
    evaluation.evaluation()
    evaluation.log_into_mlflow()


if __name__ == "__main__":
    main()
