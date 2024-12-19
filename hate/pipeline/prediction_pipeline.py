import os, re
import sys, string
import keras
import pickle
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.configuration.mongo_db_connection import MongoDBClient
from hate.components.data_transforamation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class PredictionPipeline:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictionModel", "model")
        self.tokenizer_path = os.path.join("artifacts", "PredictionModel")
        self.db_cloud = MongoDBClient()
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig,data_ingestion_artifacts=DataIngestionArtifacts)
    
    def get_best_model_from_mongodb(self) -> str:
        try:
            logging.info("Entered the get_best_model_from_mongodb method of Model Evaluation class")
            os.makedirs(self.model_path, exist_ok=True)
            self.db_cloud.load_model_from_db(target_dir=self.model_path)
            self.db_cloud.load_model_tokenizer_from_db(target_dir=self.tokenizer_path)
            logging.info("Exited the get_best_model_from_mongodb method of Model Evaluation class")
        except Exception as e:
            raise CustomException(e, sys) from e 
        
    def predict(self, text):
        logging.info("Running the predict function")
        try:
            load_model=keras.models.load_model(self.model_path)
            with open(self.tokenizer_path+'/tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            words = str(text).lower()
            words = re.sub('\[.*?\]', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            text=self.data_transformation.data_cleaning(words)
            text = [text]            
            print(text)
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            pred = load_model.predict(padded)
            if pred>0.5:
                print("hate and abusive")
                return "hate and abusive"
            else:
                print("no hate")
                return "no hate"
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self,text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            self.get_best_model_from_mongodb() 
            predicted_text = self.predict(text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e