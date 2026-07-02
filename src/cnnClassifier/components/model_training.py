import os
import tensorflow as tf
from pathlib import Path
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.vgg16 import preprocess_input
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        try:
            model_path = str(self.config.updated_base_model_path)
            if model_path.endswith('.h5') and not os.path.exists(model_path):
                model_path = model_path.replace('.h5', '.keras')
            
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Model loaded from: {model_path}")
            
        except Exception as e:
            print(f"⚠️ Load failed: {e}")
            print("🔧 Rebuilding base model...")
            self._build_model_from_scratch()
        
        # Recompile for training with Adam and lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
    
    def _build_model_from_scratch(self):
        """Rebuild VGG16 + custom layers if loading fails"""
        base_model = tf.keras.applications.VGG16(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=False
        )
        
        # Freeze layers except last few
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        # Add custom layers with regularization
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        prediction = tf.keras.layers.Dense(
            units=self.config.params_classes, 
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        
        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction)
        print("✅ Model rebuilt from scratch")
    
    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            preprocessing_function=preprocess_input,
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
                vertical_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest',
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
    
    def _get_callbacks(self):
        """Return list of callbacks to prevent overfitting"""
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.params_early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.params_reduce_lr_factor,
            patience=self.config.params_reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        
        checkpoint_path = str(self.config.trained_model_path).replace('.h5', '_best.keras')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        return [early_stop, reduce_lr, checkpoint]
    
    def _compute_class_weights(self):
        """Compute class weights to handle imbalance using actual counts"""
        class_indices = self.train_generator.class_indices
        class_weights = {}
        
        train_dir = str(self.config.training_data)
        class_counts = {}
        
        for class_name, idx in class_indices.items():
            class_path = os.path.join(train_dir, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                if count > 0:
                    class_counts[idx] = count
        
        if not class_counts:
            return {idx: 1.0 for idx in class_indices.values()}
        
        n_nonempty = len(class_counts)
        total_samples = sum(class_counts.values())
        for idx, count in class_counts.items():
            class_weights[idx] = total_samples / (n_nonempty * count)
        
        for idx in class_indices.values():
            if idx not in class_weights:
                class_weights[idx] = 1.0
        
        return class_weights
    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        callbacks = self._get_callbacks()
        class_weights = self._compute_class_weights()
        
        print(f"Class weights: {class_weights}")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.valid_generator.samples}")
        
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        self.save_model(path=self.config.trained_model_path, model=self.model)
        return history
