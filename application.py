## create a Flask app
from flask import Flask, request, render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import os

application = Flask(__name__)
app = application

### Define a route for the Flask app
@app.route('/') 
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET','POST'])   
def predict_datapoint():
    if request.method == 'GET':
        return render_template("home.html")
    data=CustomData(
        
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=float(request.form.get('writing_score')),
        writing_score=float(request.form.get('reading_score'))
        
    )
    
    ## Convert the input data into a DataFrame
    pred_df=data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    ## Create a PredictPipeline object and call the predict method
    # Note: The predict method expects a DataFrame with the same structure as the training data
    # Ensure that the input data is preprocessed in the same way as the training data
    # For example, if you used one-hot encoding or scaling during training, apply the same transformations here
    pred_df = pd.DataFrame(pred_df, index=[0]) 
    
    # initiate the PredictPipeline
    predict_pipeline=PredictPipeline()
    print("Mid Prediction")
    
    ## Call the predict method of the PredictPipeline
    # This will use the trained model and preprocessor to make predictions
    results=predict_pipeline.predict(pred_df)
    print("after Prediction")
    return render_template('home.html',results=results[0])
   
if __name__=="__main__":
    app.run(host="0.0.0.0")    