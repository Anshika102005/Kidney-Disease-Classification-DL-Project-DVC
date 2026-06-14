import os
import inspect
import tensorflow as tf
from tensorflow.keras import layers


class LoadTrainedModel:
    @staticmethod
    def _fix_input_layer():
        _orig_input = layers.InputLayer.from_config

        @classmethod
        def _fixed_input(cls, config):
            c = dict(config)
            c.pop('optional', None)
            c.pop('sparse', None)
            c.pop('ragged', None)
            if 'batch_shape' in c:
                c['batch_input_shape'] = c.pop('batch_shape')
            return _orig_input.__func__(cls, c)

        layers.InputLayer.from_config = _fixed_input

    @staticmethod
    def _fix_layer_from_config():
        _orig_layer = layers.Layer.from_config

        @classmethod
        def _fixed_layer(cls, config):
            c = dict(config)
            dtype = c.get('dtype')
            if isinstance(dtype, dict):
                try:
                    c['dtype'] = dtype.get('config', {}).get('name', 'float32')
                except Exception:
                    c['dtype'] = 'float32'
            try:
                valid = set(inspect.signature(cls.__init__).parameters)
            except Exception:
                valid = set()
            valid |= {'name', 'trainable', 'dtype'}
            filtered = {k: v for k, v in c.items() if k in valid}
            return _orig_layer.__func__(cls, filtered)

        layers.Layer.from_config = _fixed_layer

    @staticmethod
    def load(model_path=None, compile=False):
        LoadTrainedModel._fix_input_layer()
        LoadTrainedModel._fix_layer_from_config()

        if model_path is None:
            model_path = os.path.join("artifacts", "training", "trained_model.h5")

        print("TensorFlow:", tf.__version__)
        model = tf.keras.models.load_model(model_path, compile=compile)
        print("Model loaded successfully!")
        return model
