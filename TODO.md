# TODO - Reduce training time by organizing dataset + fast input pipeline

## Plan (approved)

### 1) Clean up & make dataset organization deterministic (avoid recopy)
- Update `src/cnnClassifier/components/data_ingestion.py` to:
  - Skip copying when `unzip_dir/train/<class>` and `unzip_dir/test/<class>` already exist.
  - Ensure the output structure is exactly `unzip_dir/train/<ClassName>/image.jpg` and `unzip_dir/test/<ClassName>/...`.

### 2) Replace slow ImageDataGenerator loader with tf.data + caching/prefetch
- Update `src/cnnClassifier/components/model_training.py`:
  - Use `tf.keras.utils.image_dataset_from_directory` (train/val split from the training dir).
  - Apply preprocessing via `preprocess_input` in a `map` function.
  - Add `.cache()` and `.prefetch(tf.data.AUTOTUNE)`.
  - Keep augmentation when enabled using `tf.keras.layers` transforms.
  - Remove `_compute_class_weights()` filesystem scans (compute from dataset labels once).

### 3) Update evaluation to match tf.data (optional but recommended)
- If evaluation is also slow due to generator scanning, switch `src/cnnClassifier/components/model_evaluation_mlflow.py` to tf.data with caching/prefetch.

## Followup steps
- Run `dvc repro training`
- Compare epoch time before vs after.

## Progress
- [x] Step 0: Repo analysis completed
- [x] Step 1: Update data ingestion to avoid unnecessary recopy
- [x] Step 2: Update training pipeline to tf.data with cache/prefetch
- [ ] Step 3: (Optional) Update evaluation pipeline to tf.data with cache/prefetch


