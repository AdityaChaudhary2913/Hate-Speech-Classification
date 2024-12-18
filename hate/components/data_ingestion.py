import sys
import os
import pandas as pd
from zipfile import Path
from hate.constants import *
from hate.exception import CustomException
from hate.logger import logging
from hate.data_access.phishing_data import PhishingData
from hate.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_data_into_raw_data_dir(self) -> pd.DataFrame:
        try:
            logging.info(f"Exporting data from mongodb")
            logging.info("Entered the export_data_into_raw_data_dir method of Data ingestion class")
            
            raw_batch_files_path = self.data_ingestion_config.DATA_INGESTION_DATA_DIR
            os.makedirs(raw_batch_files_path, exist_ok=True)
            
            incoming_data = PhishingData(db_name=MONGO_DATABASE_NAME)
            logging.info(f"Saving exported data into feature store file path: {raw_batch_files_path}")
            for collection_name, dataset in incoming_data.export_collections_as_dataframe():
                if collection_name == 'dataset':
                    logging.info(f"Shape of {collection_name}: {dataset.shape}")
                    # feature_store_file_path = os.path.join(raw_batch_files_path, collection_name + '.csv')
                    # logging.info(f"feature_store_file_path-----{feature_store_file_path}")
                    dataset.to_csv(raw_batch_files_path + f'/{collection_name}.csv', index=False)
            logging.info("Exited the export_data_into_raw_data_dir method of Data ingestion class")
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> Path:
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try:
            self.export_data_into_raw_data_dir()
            logging.info("Got the data from mongodb")
            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")
            return self.data_ingestion_config.DATA_INGESTION_DATA_DIR
        except Exception as e:
            raise CustomException(e, sys) from e