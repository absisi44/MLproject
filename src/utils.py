## import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
#import dill
import pickle

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

## create a function to save the object
def save_object(file_path, obj):
    """
    Save an object to a file using dill.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    
    Raises:
    - CustomException: If there is an error during saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e ,sys)

## create a function to evaluate the model
def evaluate_model(X_train, y_train, X_test, y_test, models):

    """
    Evaluate multiple regression models and return their performance metrics.
    
    Parameters:
    - X_train: Training feature set.
    - y_train: Training target variable.
    - X_test: Testing feature set.
    - y_test: Testing target variable.
    - models (dict): Dictionary of model names and their instances.
    
    Returns:
    - model_report (dict): Dictionary containing model names and their R2 scores.
    
    Raises:
    - CustomException: If there is an error during model evaluation.
    """
    try:
        model_report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = models[model_name]

            # If the model is a GridSearchCV, fit it directly
            if isinstance(model, GridSearchCV):
                model.fit(X_train, y_train)
                best_model = model.best_estimator_
                y_pred = best_model.predict(X_test)
            else:
                # Fit the model and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate R2 score
            r2_square = r2_score(y_test, y_pred)
            model_report[model_name] = r2_square

        # If hyperparameter tuning is required
        
        return model_report
    except Exception as e:
        raise CustomException(e, sys)