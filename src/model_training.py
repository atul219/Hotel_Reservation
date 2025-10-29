import os, sys
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import  *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn


logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, train_path, test_path, model_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_path = model_path
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    
    def load_and_split(self):
        try:
            logger.info(f"Loading data from {self.train_path} and {self.test_path}")
            train_df = load_data(file_path = self.train_path)
            test_df = load_data(file_path = self.test_path)

            X_train = train_df.drop(columns= ["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns= ["booking_status"])
            y_test = test_df["booking_status"]

            logger.info(f"Data splitted for model training")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"error occurred while loading data:  {e}")
            raise CustomException("error occurred while loading data", e)
        
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info(f"Initializing our model")
            lgbm_model = LGBMClassifier(
                random_state= self.random_search_params["random_state"],
            )

            logger.info(f"Starting our hyper parameter tuning")

            random_search = RandomizedSearchCV(
                            estimator= lgbm_model,
                            param_distributions= self.params_dist,
                            n_iter= self.random_search_params['n_iter'],
                            cv= self.random_search_params['cv'],
                            verbose= self.random_search_params['n_iter'],
                            n_jobs= self.random_search_params['n_jobs'],
                            random_state= self.random_search_params['random_state'],
                            scoring= self.random_search_params['scoring']
                        )
            
            logger.info(f"Starting our model training")
            random_search.fit(X_train, y_train)

            logger.info(f"Hyper Parameter tunning completed")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_     

            logger.info(f"Best parameters are: {best_params}")

            return best_lgbm_model

        except Exception as e:
            logger.error(f"error occurred while training model:  {e}")
            raise CustomException("error occurred while training model", e)  


    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info(f"Starting Data evaluation")
            y_preds = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_preds)
            precision = precision_score(y_test, y_preds)
            recall = recall_score(y_test, y_preds)
            f1 = f1_score(y_test, y_preds)

            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"F1 Score : {f1}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }


        except Exception as e:
            logger.error(f"error occurred while training model:  {e}")
            raise CustomException("error occurred while training model", e)

    def save_model(self, model):
        try:
            logger.info(f"Saving Model")
            os.makedirs(os.path.dirname(self.model_path), exist_ok= True)
            joblib.dump(model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"error occurred while saving model:  {e}")
            raise CustomException("error occurred while saving model", e)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info(f"Starting Model Training")
                logger.info(f"Starting mlflow experimentation")
                logger.info(f"Logging the training and testing dataset to MLFLOW")
                mlflow.log_artifact(self.train_path, artifact_path = "datasets")
                mlflow.log_artifact(self.test_path, artifact_path = "datasets")


                X_train, y_train, X_test, y_test = self.load_and_split()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(model = best_lgbm_model, X_test= X_test, y_test = y_test)
                self.save_model(best_lgbm_model)

                logger.info(f"Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_path)
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info(f"Model trained successfully")
                return metrics
        except Exception as e:
            logger.error(f"error occurred while running training model:  {e}")
            raise CustomException("error occurred while running training model", e)
