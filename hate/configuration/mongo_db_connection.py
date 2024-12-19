import os
import sys
import certifi
import pymongo
import gridfs
import pickle
from tensorflow.keras.models import load_model
from hate.constants import *
from hate.logger import logging
from hate.exception import CustomException

ca = certifi.where()

class MongoDBClient:
    client = None
    def __init__(self, database_name=MONGO_DATABASE_NAME):
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("MONGODB_URL") or "mongodb+srv://aditya1306:aditya1306@phishingclassifier.y2fj1x3.mongodb.net/"
                if mongo_db_url is None:
                    raise Exception("Environment key: MONGO_DB_URL is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.fs = gridfs.GridFS(self.database)
        except Exception as e:
            raise CustomException(e, sys)

    def save_model_to_db(self, model_dir, model_name='saved_model'):
        try:
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        data = f.read()
                        self.fs.put(data, filename=f"{model_name}/{os.path.relpath(file_path, model_dir)}")
        except Exception as e:
            raise CustomException(e, sys)

    def load_model_from_db(self, target_dir, model_name='saved_model'):
        try:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for grid_out in self.fs.find({'filename': {'$regex': f'^{model_name}/'}}):
                file_path = os.path.join(target_dir, os.path.relpath(grid_out.filename, model_name))
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(grid_out.read())
        except Exception as e:
            raise CustomException(e, sys)
        
    def load_model_tokenizer_from_db(self, target_dir, tokenizer_name='tokenizer'):
        try:
            try:
                tokenizer_data = self.fs.get_last_version(filename=tokenizer_name)
            except gridfs.errors.NoFile:
                logging.info("Tokenizer not found in database.")
                print("Tokenizer not found in database.")
            tokenizer_path = os.path.join(target_dir, f"{tokenizer_name}.pickle")
            try:  
                with open(tokenizer_path, 'wb') as f:
                    f.write(tokenizer_data.read())
            except Exception as e:
                logging.info("Tokenizer Cant be written.")
                print("Tokenizer Cant be written.")
        except Exception as e:
            raise CustomException(e, sys)

    def save_model_tokenizer_to_db(self, tokenizer, tokenizer_name='tokenizer'):
        try:
            temp_path = f"temp_{tokenizer_name}.pickle"
            with open(temp_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            with open(temp_path, 'rb') as f:
                tokenizer_data = f.read()
                self.fs.put(tokenizer_data, filename=tokenizer_name)
            os.remove(temp_path)
        except Exception as e:
            raise CustomException(e, sys)