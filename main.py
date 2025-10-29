from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from config.paths_config import CONFIG_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR
from utils.common_functions import read_yaml


if __name__ == "__main__":
    # data_ingestion = DataIngestion(config = read_yaml(CONFIG_PATH))
    # data_ingestion.run()

    data_processor = DataProcessor(train_path= TRAIN_FILE_PATH,
                                   test_path= TEST_FILE_PATH, 
                                   processed_dir= PROCESSED_DIR, 
                                   config_path= CONFIG_PATH)
    
    data_processor.run()