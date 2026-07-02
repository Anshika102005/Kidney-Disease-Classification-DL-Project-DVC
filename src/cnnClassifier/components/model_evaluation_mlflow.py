import os
import io
import json
import re
import zipfile
from datetime import datetime
import tensorflow as tf
import mlflow
from tensorflow.keras.applications.vgg16 import preprocess_input
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.test_data,
            shuffle=False,
            **dataflow_kwargs
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

        if path.endswith(".keras"):
            try:
                import keras
                if keras.__version__.startswith("2."):
                    with zipfile.ZipFile(path, "r") as zf:
                        config = json.loads(zf.read("config.json"))
                        raw_weights = zf.read("model.weights.h5")

                    config = self._preprocess_config(config)
                    config["compile_config"] = None

                    orig_input = self._patch_input_layer()
                    orig_policy = self._patch_dtype_policy()
                    try:
                        from keras.saving.serialization_lib import deserialize_keras_object
                        from keras.saving.saving_lib import ObjectSharingScope

                        with ObjectSharingScope():
                            self.model = deserialize_keras_object(
                                config,
                                custom_objects={"Functional": tf.keras.Model},
                                safe_mode=True,
                            )

                        self._set_weights_from_h5(self.model, raw_weights)
                    finally:
                        self._restore_input_layer(orig_input)
                        self._restore_dtype_policy(orig_policy)

                    logger.info("Model loaded from .keras (Keras 2 compatible mode)")
                else:
                    self.model = tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                logger.warning(f"Keras 2 load failed, trying default: {e}")
                self.model = tf.keras.models.load_model(path, compile=False)
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
            logger.info("Scores not found, running evaluation first...")
            self.evaluation()
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        scores_path = os.path.join(os.getcwd(), "scores.json")
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
        logger.info(f"Scores saved to {scores_path}")

        missing_credentials = [
            name for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
            if not os.getenv(name)
        ]
        if missing_credentials:
            logger.warning(
                "Skipping MLflow logging because credentials are missing: "
                + ", ".join(missing_credentials)
            )
            return

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
