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
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param_grid: dict = None):
    try:
        model_report = {}

        for model_name, model in models.items():
            params = param_grid.get(model_name) if param_grid else None

            print(f"Evaluating model: {model_name}")
            if params:
                print(f"Hyperparameter tuning with: {params}")
                model = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=2)
                model.fit(X_train, y_train)
                final_model = model.best_estimator_
            else:
                print("No hyperparameter tuning")
                model.fit(X_train, y_train)
                final_model = model

            y_pred = final_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[model_name] = r2

        return model_report

    except Exception as e:
        raise CustomException(e, sys)
