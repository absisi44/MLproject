'''
Data Ingestion Component
This component is responsible for ingesting data from various sources.
It can handle CSV files, databases, and other data formats as needed.
'''
## Import necessary libraries
import os 
from pyexpat import model
import sys
import pandas as pd
from sklearn.model_selection import train_test_split 

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrianer

## Create a dataclass for Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    
     '''
     This creates a simple class to hold the file paths for:

    the raw data,the training data ,the test data 
    All files will be saved inside the artifacts folder
    '''
     train_data_path: str = os.path.join('artifacts','train.csv')
     test_data_path: str = os.path.join('artifacts','test.csv')
     raw_data_path: str = os.path.join('artifacts','data.csv')
     
## Create a Data Ingestion class
class DataIngestion:
    
    def __init__(self):
         self.ingestion_config = DataIngestionConfig()   
           
     
    ## initiate_data_ingestion Function 
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
         
        try:
            
            # Read the data from the source
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Data read successfully from the source')
             # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train Test Split initiated')
        
            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test data to the specified paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info('Data ingestion completed successfully')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys) 
            
             
            
                        
                
if __name__== "__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)    
       
    model_trainer = ModelTrianer()
    print(model_trainer.initiate_model_training(train_arr, test_arr))
    
    logging.info("Model training completed successfully")

#  
 

             
    



