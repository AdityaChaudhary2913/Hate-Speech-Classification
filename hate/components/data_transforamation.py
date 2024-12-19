import os
import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from hate.logger import logging 
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
        
    def read_data_from_data_ingestion_dir(self):
        try:
            logging.info("Entered into the read_data_from_data_ingestion_dir function")
            data=pd.read_csv(self.data_ingestion_artifacts.data_file_path)
            # data = data.sample(frac=0.05) # for testing purpose
            logging.info("Exited the read_data_from_data_ingestion_dir function")
            return data 
        except Exception as e:
            raise CustomException(e,sys) from e 

    def data_cleaning(self, words):
        try:
            logging.info("Entered into the concat_data_cleaning function")
            original_words = words
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            words = [word for word in words.split(' ') if words not in stopword]
            words=" ".join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words=" ".join(words)
            if not words:
                logging.warning("Cleaned content was empty, reverting to original content")
                words = original_words
            logging.info("Exited the concat_data_cleaning function")
            return words 
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")
            df = self.read_data_from_data_ingestion_dir()
            df[self.data_transformation_config.CONTENT]=df[self.data_transformation_config.CONTENT].apply(self.data_cleaning)

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH,index=False,header=True)

            data_transformation_artifact = DataTransformationArtifacts(transformed_data_path = self.data_transformation_config.TRANSFORMED_FILE_PATH)
            logging.info("returning the DataTransformationArtifacts")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
