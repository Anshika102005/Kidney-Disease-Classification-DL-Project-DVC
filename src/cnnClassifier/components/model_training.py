import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        # ✅ FIX: Load with compile=False to avoid deserialization issues
        try:
            # Try loading .keras first
            model_path = str(self.config.updated_base_model_path)
            if model_path.endswith('.h5') and not os.path.exists(model_path):
                model_path = model_path.replace('.h5', '.keras')
            
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Model loaded from: {model_path}")
            
        except Exception as e:
            print(f"⚠️ Load failed: {e}")
            print("🔧 Rebuilding base model...")
            self._build_model_from_scratch()
        
        # Recompile for training
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
    
    def _build_model_from_scratch(self):
        """Rebuild VGG16 + custom layers if loading fails"""
        # VGG16 base
        base_model = tf.keras.applications.VGG16(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=False
        )
        
        # Freeze layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom layers
        flatten = tf.keras.layers.Flatten()(base_model.output)
        prediction = tf.keras.layers.Dense(units=2, activation='softmax')(flatten)
        
        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction)
        print("✅ Model rebuilt from scratch")
    
    def train_valid_generator(self):
        # Your existing code...
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
        
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
        
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path = str(path)
        if path.endswith('.h5'):
            path = path.replace('.h5', '.keras')
        model.save(path)
    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )
        
        self.save_model(path=self.config.trained_model_path, model=self.model)