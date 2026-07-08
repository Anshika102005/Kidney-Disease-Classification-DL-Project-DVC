import os
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.applications.vgg16 import preprocess_input

from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        try:
            model_path = str(self.config.updated_base_model_path)
            if model_path.endswith(".h5") and not os.path.exists(model_path):
                model_path = model_path.replace(".h5", ".keras")

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
            metrics=["accuracy"],
        )

    def _build_model_from_scratch(self):
        """Rebuild VGG16 + custom layers if loading fails"""
        base_model = tf.keras.applications.VGG16(
            input_shape=(224, 224, 3),
            weights="imagenet",
            include_top=False,
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
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )(x)

        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction)
        print("✅ Model rebuilt from scratch")

    def _build_augmentation(self):
        # Keep augmentation lightweight; VGG16 preprocess_input happens after augmentation.
        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomTranslation(0.2, 0.2),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomContrast(0.2),
            ],
            name="augmentation",
        )

    def train_valid_generator(self):
        image_size_hw = self.config.params_image_size[:2]
        batch_size = self.config.params_batch_size

        # Build datasets directly from directory (much faster than ImageDataGenerator).
        # We treat training_data as the root containing class subfolders.
        train_root = str(self.config.training_data)

        val_split = 0.20
        seed = 1337

        # Train dataset with augmentation
        full_train_ds = tf.keras.utils.image_dataset_from_directory(
            train_root,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size_hw,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            validation_split=val_split,
            subset="training",
        )

        # Validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_root,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size_hw,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            validation_split=val_split,
            subset="validation",
        )

        aug = self._build_augmentation() if self.config.params_is_augmentation else None

        def preprocess_train(images, labels):
            images = tf.cast(images, tf.float32)
            if aug is not None:
                images = aug(images, training=True)
            images = preprocess_input(images)
            return images, labels

        def preprocess_val(images, labels):
            images = tf.cast(images, tf.float32)
            images = preprocess_input(images)
            return images, labels

        autotune = tf.data.AUTOTUNE

        # Cache speeds up repeated epochs by avoiding repeated decoding/resize work.
        # Note: cache(None) caches in memory; if dataset is huge, caching to a file is better.
        full_train_ds = (
            full_train_ds.map(preprocess_train, num_parallel_calls=autotune)
            .cache()
            .prefetch(autotune)
        )
        val_ds = (
            val_ds.map(preprocess_val, num_parallel_calls=autotune)
            .cache()
            .prefetch(autotune)
        )

        self.train_generator = full_train_ds
        self.valid_generator = val_ds

        # Store for logging
        self.steps_per_epoch = None  # Keras will infer from dataset
        self.validation_steps = None

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path = str(path)
        if path.endswith(".h5"):
            path = path.replace(".h5", ".keras")
        model.save(path)

    def _get_callbacks(self):
        """Return list of callbacks to prevent overfitting"""
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.params_early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.config.params_reduce_lr_factor,
            patience=self.config.params_reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        )

        checkpoint_path = str(self.config.trained_model_path).replace(".h5", "_best.keras")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )

        return [early_stop, reduce_lr, checkpoint]

    def train(self):
        callbacks = self._get_callbacks()

        print("Training with tf.data (cache + prefetch) ...")
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=callbacks,
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)
        return history

