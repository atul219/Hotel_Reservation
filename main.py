from src.data_ingestion import DataIngestion
from config.paths_config import CONFIG_PATH
from utils.common_functions import read_yaml


if __name__ == "__main__":
    data_ingestion = DataIngestion(config = read_yaml(CONFIG_PATH))
    data_ingestion.run()