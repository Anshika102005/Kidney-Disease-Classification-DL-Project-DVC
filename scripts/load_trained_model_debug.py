import tensorflow as tf
import inspect
from tensorflow.keras import layers

# ===== 1. InputLayer fix =====
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

# ===== 2. Layer from_config fix (Dense, Conv2D, etc.) =====
_orig_layer = layers.Layer.from_config

@classmethod
def _fixed_layer(cls, config):
    c = dict(config)

    # dtype dict → string
    dtype = c.get('dtype')
    if isinstance(dtype, dict):
        try:
            c['dtype'] = dtype.get('config', {}).get('name', 'float32')
        except Exception:
            c['dtype'] = 'float32'

    # keep only valid init kwargs
    try:
        valid = set(inspect.signature(cls.__init__).parameters)
    except Exception:
        valid = set()

    valid |= {'name', 'trainable', 'dtype'}
    filtered = {k: v for k, v in c.items() if k in valid}

    return _orig_layer.__func__(cls, filtered)

layers.Layer.from_config = _fixed_layer

# ===== 3. Load model =====
print('TensorFlow:', tf.__version__)

model_path = r"c:\Users\LENOVO\Kidney-Disease-Classification-DL-Project-DVC\artifacts\training\trained_model.h5"

model = tf.keras.models.load_model(model_path, compile=False)

print('Model loaded successfully!')
model.summary()
