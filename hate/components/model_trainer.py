import os 
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts,DataTransformationArtifacts
from hate.ml.model import ModelArchitecture


class ModelTrainer:
    def __init__(self,data_transformation_artifacts: DataTransformationArtifacts,model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
    
    def spliting_data(self,csv_path):
        try:
            logging.info("Entered the spliting_data function")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            # df = df.sample(frac=0.01) # for testing purpose
            logging.info("Splitting the data into x and y")
            x = df[CONTENT]
            y = df[LABEL]
            logging.info("Applying train_test_split on the data")
            x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=self.model_trainer_config.VALIDATION_SPLIT,random_state = self.model_trainer_config.RANDOM_STATE)
            print(len(x_train),len(y_train))
            print(len(x_test),len(y_test))
            print(type(x_train),type(y_train))
            logging.info("Exited the spliting the data function")
            return x_train,x_test,y_train,y_test
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def tokenizing(self,x_train):
        try:
            logging.info("Applying tokenization on the data")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences,maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f" The sequence matrix is: {sequences_matrix}")
            return sequences_matrix,tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            logging.info("Entered the initiate_model_trainer function ")
            x_train,x_test,y_train,y_test = self.spliting_data(csv_path=self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()   
            model = model_architecture.get_model()
            logging.info(f"Xtrain size is : {x_train.shape}")
            logging.info(f"Xtest size is : {x_test.shape}")
            sequences_matrix,tokenizer =self.tokenizing(x_train)
            logging.info("Entered into model training")
            early_stop = EarlyStopping(monitor=self.model_trainer_config.MONITOR, patience=self.model_trainer_config.PATIENCE, restore_best_weights=True)
            model.fit(sequences_matrix, y_train, 
                        batch_size=self.model_trainer_config.BATCH_SIZE, 
                        epochs = self.model_trainer_config.EPOCH, 
                        validation_split=self.model_trainer_config.VALIDATION_SPLIT, 
                        callbacks=[early_stop],
                        verbose=1
                        )
            # model.fit(sequences_matrix, y_train, batch_size=32, epochs=1, validation_split=0.2) # for testing purpose
            logging.info("Model training finished")
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)
            with open(self.model_trainer_config.TRAINED_TOKENIZER_PATH, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH, save_format='tf')
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)
            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                trained_tokenizer_path = self.model_trainer_config.TRAINED_TOKENIZER_PATH,
                x_test_path = self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path = self.model_trainer_config.Y_TEST_DATA_PATH)
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e