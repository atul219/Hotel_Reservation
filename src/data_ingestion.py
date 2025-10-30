import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_FILE_PATH, RAW_DIR, TRAIN_FILE_PATH, TEST_FILE_PATH
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data Ingestion started with {self.bucket_name} and file name {self.file_name}")

    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Raw file is successfully downloaded to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error("Error while downloading csv file")
            raise CustomException("Failed to download csv file from gcp bucket", e)
        
    
    def split_data(self):
        try:
            logger.info(f"Starting the splitting process")
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data, test_size= 1 - self.train_test_ratio, random_state= 42)

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"Train and Test data saved to {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
        
        except Exception as e:
            logger.error("Error while splitting csv file")
            raise CustomException("Failed to split data into training and test", e)

    def run(self):
        try:
            logger.info(f"Starting data ingestion process")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info(f"Data ingestion completed")
        except Exception as e:
            logger.error(f"Custiom exception: {str(e)}")
            raise CustomException("Error in data ingestion", e)

            
