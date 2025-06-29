## import necessary libraries 
import sys
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object

## create a predict pipeline predict class
class PredictPipeline:
    def __init__(self):
        pass
    
    ## Create a function to predict the output based on the input data
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            
            # Clamp and round the prediction
            predictions = [ round(max(0, min(100, pred))) for pred in predictions]
            
            return predictions
        except Exception as e:
            raise CustomException(e,sys)


## Create a CustomData class to map the input data
class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender=gender
        
        self.race_ethnicity=race_ethnicity
        
        self.parental_level_of_education=parental_level_of_education
        
        self.lunch=lunch
        
        self.test_preparation_course=test_preparation_course
        
        self.reading_score=reading_score
        
        self.writing_score=writing_score
    
    ## Create a function to convert the input data into a DataFrame
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'gender':self.gender,
                'race_ethnicity':self.race_ethnicity,
                'parental_level_of_education':self.parental_level_of_education,
                'lunch':self.lunch,
                'test_preparation_course':self.test_preparation_course,
                'reading_score':self.reading_score,
                'writing_score':self.writing_score
            }
            
            return pd.DataFrame(custom_data_input_dict, index=[0])
        except Exception as e :
            raise CustomException(e,sys)    