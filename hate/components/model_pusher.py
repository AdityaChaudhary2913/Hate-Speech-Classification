import sys, os
import pickle
from hate.logger import logging
from hate.exception import CustomException
from hate.configuration.mongo_db_connection import MongoDBClient
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifacts

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config
        self.db_client = MongoDBClient()

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        logging.info("Entered the initiate_model_pusher method of ModelPusher class")
        try:
            # Pushing the trained model to MongoDB
            model_dir = os.path.join(self.model_pusher_config.TRAINED_MODEL_PATH, self.model_pusher_config.MODEL_NAME)
            self.db_client.save_model_to_db(model_dir)
            logging.info(f"Pushed model to MongoDB.")

            # Pushing the tokenizer to MongoDB
            tokenizer_path = self.model_pusher_config.TRAINED_TOKENIZER_PATH
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            self.db_client.save_model_tokenizer_to_db(tokenizer)
            logging.info(f"Pushed tokenizer to MongoDB.")

            logging.info("Exited the initiate_model_pusher method successfully.")
        except Exception as e:
            logging.error("An error occurred while pushing the model and tokenizer to MongoDB.", exc_info=True)
            raise CustomException(e, sys) from e