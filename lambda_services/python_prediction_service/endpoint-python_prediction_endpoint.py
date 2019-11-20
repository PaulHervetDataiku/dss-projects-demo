from  dataiku.apinode.predict.predictor import ClassificationPredictor
import pandas as pd
import pickle

class MyPredictor(ClassificationPredictor):

    def __init__(self, data_folder):
        self.data_folder = data_folder

    def predict(self, features_df):       
        # Load model from file
        folder_path = self.data_folder
        pkl_filename = folder_path + "/pickle_model.pkl"
        
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)

        
        # Get predictions
        #predictions = pd.Series(["good", "fair", "poor", "good", "poor"])
        predictions = pd.Series(pickle_model.predict(features_df))
        
        
        # Get probas (optional)
        #probas = pd.DataFrame({
        #    'proba_good': pd.Series([.9, .6, .2, .7, .2]),
        #    'proba_fair': pd.Series([.2, .7, .3, .3, .3]),
        #    'proba_poor': pd.Series([.2, .6, .6, .3, .9])
        #})
        #probas = 

        return (predictions) #, probas
