## import necessary libraries 
import os 
import sys
import pandas as pd 
import numpy as np 

from dataclasses import dataclass
from seaborn import categorical
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    

## The Main Work (DataTransformation Class)
class DataTransformation:
    def __init__(self):
        self.data_transforming_config = DataTransformationConfig()
    
    # Function to get the preprocessor object
    def get_data_transformer_object(self):
        
        """This makes a "machine" that will clean and prepare the data:

            -For number columns (writing_score, reading_score):

            1. Fills in missing values with the median (middle value)

            2. Scales the numbers to a standard range

            -For category columns (gender, race, education etc.):

            1. Fills in missing values with the most common value

            2. Converts categories to numbers (like making "male/female" into 0/1)

            3. Scales these numbers too"""
            
        try:
            numerical_columns =["writing_score","reading_score"] 
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",                                                            
                "lunch",
                "test_preparation_course"  
            ] 
            
            Num_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler" , StandardScaler())
                ]
            )
            
            Cat_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    #("scaler" , StandardScaler(with_mean=False)),
                    ("onehotencoder", OneHotEncoder())
                    
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            # Preprocessor object
            preprocessor = ColumnTransformer([
                ("num_pipeline", Num_pipeline, numerical_columns),
                ("cat_pipeline", Cat_pipeline, categorical_columns)
            ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys) 
    
    ## Applying the Transformation (initiate_data_transformation)
    def initiate_data_transformation(self, train_path, test_path):
        """This function does the actual cleaning and preparing of the data:
            1. Loads the training and test data from files

            2. Cleans and prepares the data using the preprocessor machine

            3. Saves this prepared data to files for later use
            4. Returns the paths to the saved files"""    
        try:
            # Load the training and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Loaded training and test data successfully")
            logging.info("Obtaining preprocessing object")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            # set the target column name 
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            # Separate features and target variable
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            # Apply the preprocessor to the training and test data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            # Convert target variables to numpy arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saved preprocessing object")
            
            save_object(
                file_path = self.data_transforming_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transforming_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)            
    
    