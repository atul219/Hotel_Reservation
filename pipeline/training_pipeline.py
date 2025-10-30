from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from config.paths_config import CONFIG_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH
from utils.common_functions import read_yaml



if __name__ == "__main__":

    data_ingestion = DataIngestion(config = read_yaml(CONFIG_PATH))
    data_ingestion.run()

    data_processor = DataProcessor(train_path= TRAIN_FILE_PATH,
                                   test_path= TEST_FILE_PATH, 
                                   processed_dir= PROCESSED_DIR, 
                                   config_path= CONFIG_PATH)
    
    data_processor.run()

    model_trainer = ModelTraining(train_path= PROCESSED_TRAIN_DATA_PATH,
                                  test_path= PROCESSED_TEST_DATA_PATH,
                                  model_path= MODEL_OUTPUT_PATH)
    
    metrics = model_trainer.run()
    print(metrics)