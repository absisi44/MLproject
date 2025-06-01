## import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
#import dill
import pickle

from src.exception import CustomException

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

