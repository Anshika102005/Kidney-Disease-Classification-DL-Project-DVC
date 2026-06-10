import os
import zipfile
import shutil
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Copy local dataset file to artifacts directory
        (Replaces Google Drive download for local machine usage)
        '''
        try:
            source_path = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            
            # Create artifacts directory
            zip_dir = os.path.dirname(zip_download_dir)
            if zip_dir:
                os.makedirs(zip_dir, exist_ok=True)
            
            # Check if source is a local file path
            if source_path == "local":
                if not os.path.exists(zip_download_dir):
                    raise FileNotFoundError(
                        f"Local file not found: {zip_download_dir}\n"
                        f"Please check your config.yaml path."
                    )
                logger.info(f"Using local file: {zip_download_dir}")
                return str(zip_download_dir)
            
            # If source_URL is an actual file path
            elif os.path.exists(source_path):
                logger.info(f"Copying data from {source_path} to {zip_download_dir}")
                
                if not os.path.exists(zip_download_dir):
                    shutil.copy(source_path, zip_download_dir)
                    logger.info(f"Copied successfully! Size: {get_size(Path(zip_download_dir))}")
                else:
                    logger.info(f"File already exists at: {zip_download_dir}")
                
                return str(zip_download_dir)
            
            else:
                raise FileNotFoundError(
                    f"Source not found: {source_path}\n"
                    f"Please set source_URL to 'local' or a valid file path in config.yaml"
                )

        except Exception as e:
            logger.error(f"Error in download_file: {e}")
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            
            if not os.path.exists(self.config.local_data_file):
                raise FileNotFoundError(
                    f"ZIP file not found: {self.config.local_data_file}\n"
                    f"Run download_file() first."
                )
            
            if not zipfile.is_zipfile(self.config.local_data_file):
                raise zipfile.BadZipFile(
                    f"File is not a valid ZIP: {self.config.local_data_file}\n"
                    f"File size: {os.path.getsize(self.config.local_data_file)} bytes"
                )
            
            logger.info(f"Extracting {self.config.local_data_file} to {unzip_path}")
            
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            
            logger.info(f"Extraction completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in extract_zip_file: {e}")
            raise e