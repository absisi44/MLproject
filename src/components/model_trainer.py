## import necessary libraries 
import os
import sys
import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException
from src.utils import save_object, evaluate_models
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dataclasses import dataclass

## Create a dataclass for model trainer configuration
@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    
##create a class for model trainer
class ModelTrianer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 
    
    ## create a function to initiate model training
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
            )
            
            ## create a dictionary of models
            models ={
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            ## create a param grid for hyperparameter tuning
            param_grid = {
                "Linear Regression": {},
                "KNeighbors Regressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
                "Random Forest Regressor": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20],
                },
                "Gradient Boosting Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },
                "XGBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },
                "CatBoost Regressor": {},
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },
            }
            
            logging.info("Model training started")
            
            
            ## create a dictionary to store model reports
            model_reports:dict = evaluate_models(X_train= X_train,
                                                y_train= y_train,
                                                X_test= X_test,
                                                y_test= y_test,
                                                models=models,param_grid=param_grid)
            best_model_score = max(sorted(model_reports.values()))
            best_model_name = list(model_reports.keys())[
                list(model_reports.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            ## save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            mean_absolute_error_value = mean_absolute_error(y_test, predicted)
            mean_squared_error_value = mean_squared_error(y_test, predicted)
            
            logging.info(f"R2 Score: {r2_square}")
            logging.info(f"Mean Absolute Error: {mean_absolute_error_value}")
            logging.info(f"Mean Squared Error: {mean_squared_error_value}")
            
            return (
                best_model_name,
                r2_square,
                mean_absolute_error_value,
                mean_squared_error_value
            )
        
        except Exception as e:
            raise CustomException(e, sys)
              