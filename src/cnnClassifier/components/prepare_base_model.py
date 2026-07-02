import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, dropout_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
        
        pool = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        bn = tf.keras.layers.BatchNormalization()(pool)
        dropout = tf.keras.layers.Dropout(dropout_rate)(bn)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(dropout)
        
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        params = self.config
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=params.params_classes,
            freeze_all=params.params_freeze_all if hasattr(params, 'params_freeze_all') else False,
            freeze_till=params.params_freeze_till if hasattr(params, 'params_freeze_till') else None,
            learning_rate=params.params_learning_rate,
            dropout_rate=params.params_dropout if hasattr(params, 'params_dropout') else 0.5
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path = str(path)
        if path.endswith('.h5'):
            path = path.replace('.h5', '.keras')
        
        model.save(path)
        print(f"Model saved to: {path}")