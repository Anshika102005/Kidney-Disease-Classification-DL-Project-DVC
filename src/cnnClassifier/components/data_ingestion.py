import os
import random
import shutil
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Copy the classification dataset from source to artifacts directory.
        """
        try:
            source_path = self.config.source_URL
            dest_path = self.config.local_data_file
            dest_dir = os.path.dirname(dest_path) if os.path.dirname(dest_path) else "."
            os.makedirs(dest_dir, exist_ok=True)

            if source_path == "local":
                if not os.path.exists(dest_path):
                    raise FileNotFoundError(
                        f"Dataset not found at: {dest_path}\n"
                        f"Place CT-KIDNEY-DATASET at this location or set source_URL "
                        f"in config.yaml to the dataset path."
                    )
                logger.info(f"Using local dataset: {dest_path}")
                return str(dest_path)

            elif os.path.exists(source_path):
                if os.path.exists(dest_path):
                    logger.info(f"Dataset already exists at: {dest_path}")
                else:
                    logger.info(f"Copying dataset from {source_path} to {dest_path}")
                    if os.path.isdir(source_path):
                        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source_path, dest_path)
                    logger.info(f"Copied successfully!")
                return str(dest_path)

            else:
                raise FileNotFoundError(
                    f"Source not found: {source_path}\n"
                    f"Please set source_URL to 'local' or a valid path in config.yaml"
                )

        except Exception as e:
            logger.error(f"Error in download_file: {e}")
            raise e

    @staticmethod
    def _copy_class_split(src_dir, dest_dir, class_names):
        """Copy images from src_dir/<class>/ to dest_dir/<class>/"""
        for class_name in class_names:
            src_class = src_dir / class_name
            if src_class.exists():
                dest_class = dest_dir / class_name
                dest_class.mkdir(parents=True, exist_ok=True)
                for img in src_class.iterdir():
                    if img.is_file():
                        shutil.copy2(str(img), str(dest_class / img.name))

    def organize_classification_data(self):
        """
        Organize the classification dataset for the training pipeline.
        If the dataset has train/valid/test splits, uses them directly
        (merging valid into train). Otherwise splits the flat class folders
        into train (70%), valid (15%), test (15%).
        Output: <unzip_dir>/train/{class}/  and  <unzip_dir>/test/{class}/
        """
        try:
            dataset_path = Path(self.config.local_data_file)
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at: {dataset_path}. Run download_file() first."
                )

            unzip_path = Path(self.config.unzip_dir)
            train_dest = unzip_path / "train"
            test_dest = unzip_path / "test"

            class_names = ["Cyst", "Normal", "Stone", "Tumor"]

            train_src = dataset_path / "train"
            valid_src = dataset_path / "valid"
            test_src = dataset_path / "test"

            # Pre-split dataset: copy splits as-is (merge valid into train)
            if train_src.exists():
                self._copy_class_split(train_src, train_dest, class_names)
                if valid_src.exists():
                    self._copy_class_split(valid_src, train_dest, class_names)
                    logger.info("Merged valid split into train")
                if test_src.exists():
                    self._copy_class_split(test_src, test_dest, class_names)
            else:
                # Flat dataset: split into train (70%) / valid (15%) / test (15%)
                logger.info("No pre-existing splits found, creating 70/15/15 split...")
                for class_name in class_names:
                    class_dir = dataset_path / class_name
                    if not class_dir.exists():
                        continue
                    images = [f for f in class_dir.iterdir() if f.is_file()]
                    if not images:
                        continue
                    random.shuffle(images)
                    n_test = max(1, int(len(images) * 0.15))
                    n_valid = max(1, int(len(images) * 0.15))

                    test_imgs = images[:n_test]
                    valid_imgs = images[n_test:n_test + n_valid]
                    train_imgs = images[n_test + n_valid:]

                    for img in train_imgs + valid_imgs:
                        (train_dest / class_name).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(img), str(train_dest / class_name / img.name))

                    for img in test_imgs:
                        (test_dest / class_name).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(img), str(test_dest / class_name / img.name))

                logger.info("Dataset split into train/valid/test and valid merged into train")

            for split_name, split_dir in [("train", train_dest), ("test", test_dest)]:
                counts = {}
                if split_dir.exists():
                    for class_dir in sorted(split_dir.iterdir()):
                        if class_dir.is_dir():
                            n = len(list(class_dir.iterdir()))
                            counts[class_dir.name] = n
                logger.info(f"Classification data ({split_name}): {counts}")

            logger.info("Dataset organization completed successfully!")

        except Exception as e:
            logger.error(f"Error organizing classification data: {e}")
            raise e
