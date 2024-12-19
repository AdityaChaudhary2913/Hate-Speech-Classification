import os
import sys
import keras
import pickle
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.constants import *
from hate.ml.model import ModelArchitecture
from hate.configuration.mongo_db_connection import MongoDBClient
from keras.preprocessing.text import Tokenizer
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,model_trainer_artifacts: ModelTrainerArtifacts,data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.db_cloud = MongoDBClient()

    def get_best_model_from_mongodb(self) -> str:
        try:
            logging.info("Entered the get_best_model_from_mongodb method of Model Evaluation class")
            os.makedirs(self.model_evaluation_config.MODEL_EVALUATION_MODEL_DIR, exist_ok=True)
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)
            model_file_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH, self.model_evaluation_config.MODEL_NAME)
            self.db_cloud.load_model_from_db(target_dir=model_file_path)
            self.db_cloud.load_model_tokenizer_from_db(target_dir=self.model_evaluation_config.BEST_MODEL_DIR_PATH)
            logging.info("Exited the get_best_model_from_mongodb method of Model Evaluation class")
            return model_file_path 
        except Exception as e:
            raise CustomException(e, sys) from e 
        
    def evaluate(self):
        try:
            logging.info("Entering into to the evaluate function of Model Evaluation class")
            print(self.model_trainer_artifacts.x_test_path)
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path,index_col=0)
            print(x_test)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path,index_col=0)
            with open(self.model_trainer_artifacts.trained_tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            load_model=keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            x_test = x_test['content'].astype(str)
            x_test = x_test.squeeze()
            y_test = y_test.squeeze()
            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences,maxlen=MAX_LEN)
            print(f"----------{test_sequences_matrix}------------------")
            print(f"-----------------{x_test.shape}--------------")
            print(f"-----------------{y_test.shape}--------------")
            accuracy = load_model.evaluate(test_sequences_matrix,y_test)
            logging.info(f"Test accuracy is {accuracy}")
            print(f"Test accuracy is {accuracy}")
            return accuracy
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        logging.info("Initiate Model Evaluation")
        try:
            logging.info("Loading currently trained model")
            trained_model=keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            with open(self.model_trainer_artifacts.trained_tokenizer_path, 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            trained_model_accuracy = self.evaluate()
            logging.info("Fetch best model from gcloud storage")
            best_model_path = self.get_best_model_from_mongodb()
            logging.info("Check is best model present in the gcloud storage or not ?")
            if os.path.isfile(os.path.join(best_model_path, 'saved_model.pb')) is False:
                is_model_accepted = True
                logging.info("mongodb storage model is false and currently trained model accepted is true")
            else:
                logging.info("Load best model fetched from gcloud storage")
                best_model=keras.models.load_model(best_model_path)
                best_model_accuracy= self.evaluate()
                print(f"Best model accuracy is {best_model_accuracy}")
                print(f"Trained model accuracy is {trained_model_accuracy}")
                logging.info("Comparing loss between best_model_loss and trained_model_loss ? ")
                if best_model_accuracy > trained_model_accuracy:
                    is_model_accepted = True
                    logging.info("Trained model not accepted")
                else:
                    is_model_accepted = False
                    logging.info("Trained model accepted")
            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e