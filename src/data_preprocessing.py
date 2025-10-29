import os, sys
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(file_path= config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    
    def preprocess_data(self, df):
        try:
            logger.info(f"Starting Data Processing Step")
            df.drop(columns = ["Unnamed: 0", "Booking_ID"], inplace= True)
            df.drop_duplicates(inplace= True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info(f"Applying label encoding")
            
            label_encoder = LabelEncoder()
            mappings = {}
            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label: code for label, code in zip (label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
            
            logger.info(f"Label Mappings are: ")
            for col, mapping in mappings.items():
                logger.info(f"{col} : {mapping}")

            logger.info(f"Handling Skewness")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x: x.skew())

            for column in skewness[skewness >  skew_threshold].index:
                df[column] = np.log1p(df[column])
            
            return df

        except Exception as e:
            logger.error(f"Error occurred during pre process step {e}")
            raise CustomException("Error occurred during pre process step", e)
        
    
    def balance_data(self, df):
        try:
            logger.info(f"handling Imbalanced data")
            X= df.drop(columns= "booking_status")
            y = df["booking_status"]
            smote = SMOTE(random_state= 42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns = X.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info(f"Data balanced sucessfully")

            return balanced_df
        except Exception as e:
            logger.error(f"Error occurred during balancing data step")
            raise CustomException("Error occurred during balancing data step", e) 
        
    
    def feature_selection(self, df):
        try:
            logger.info(f"Feature Selection")
            X = df.drop(columns= ["booking_status"])
            y = df["booking_status"]

            model = RandomForestClassifier(random_state= 42)
            model.fit(X, y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                                    "features" :  X.columns,
                                    "importance": feature_importance
                                })
            
            top_features_importance_df = feature_importance_df.sort_values(by = "importance", ascending= False)
            num_features_to_select = self.config["data_processing"]["no_of_features"]
            top_features = top_features_importance_df["features"].head(num_features_to_select).values
            top_features_df = df[top_features.tolist() + ["booking_status"]]
            
            logger.info(f"top features: {top_features}")
            logger.info(f"Feature Selection done")

            return top_features_df
        
        except Exception as e:
            logger.error(f"Error occurred during feature selection step")
            raise CustomException("Error occurred during feature selection step", e)
        
    
    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving data in processed folder")
            df.to_csv(file_path, index = False)
            logger.info(f"Data saved")
        except Exception as e:
            logger.error(f"Error occurred during saving data")
            raise CustomException("Error occurred during saving data", e)
        

    def run(self):
        try:
            logger.info(f"Starting data Preprocessing")
            train_data = load_data(self.train_path)
            test_data = load_data(self.test_path)

            train_df = self.preprocess_data(train_data)
            test_df = self.preprocess_data(test_data)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(df = train_df, file_path= PROCESSED_TRAIN_DATA_PATH)
            self.save_data(df = test_df, file_path= PROCESSED_TEST_DATA_PATH)

            
            logger.info(f"Data preprocessing completed")
        except Exception as e:
            logger.error(f"Error occurred during preprocessing data")
            raise CustomException("Error occurred during preprocessing data", e)

