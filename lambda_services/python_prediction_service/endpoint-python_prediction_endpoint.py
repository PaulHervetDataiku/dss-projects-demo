
from  dataiku.apinode.predict.predictor import ClassificationPredictor
import pandas as pd
import pickle

class MyPredictor(ClassificationPredictor):

    def __init__(self, data_folder):
        self.data_folder = data_folder

    def predict(self, features_df):


        print "Features DataFrame %s" % type(features_df)

        print("--------------------")
        print("--------------------")
        print("--------------------")
        print(type(features_df)
        print("--------------------")
        print("--------------------")
        print("--------------------")
        
        
        
        folder_path = self.data_folder
        pkl_filename = folder_path + "/pickle_model.pkl"
        
        # Load from file
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)

        
        
        # predictions, one per record (features_df row)
        #predictions = pd.Series(["good", "fair", "poor", "good", "poor"])
        predictions = pd.Series(pickle_model.predict(features_df))
        
        # optional probas for each class (may be None or a DataFrame with one column per class)
        #probas = pd.DataFrame({
        #    'proba_good': pd.Series([.9, .6, .2, .7, .2]),
        #    'proba_fair': pd.Series([.2, .7, .3, .3, .3]),
        #    'proba_poor': pd.Series([.2, .6, .6, .3, .9])
        #})
        #probas = 

        return (predictions) #, probas
