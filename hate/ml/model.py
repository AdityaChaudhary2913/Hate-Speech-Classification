# Creating model architecture.
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import LSTM,Dense,Embedding,SpatialDropout1D
from hate.constants import *

class ModelArchitecture:
    def __init__(self):
        pass
    
    def get_model(self):
        model = Sequential()
        model.add(Embedding(MAX_WORDS, 100,input_length=MAX_LEN))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1,activation=ACTIVATION))
        model.compile(loss=LOSS,optimizer=RMSprop(),metrics=METRICS)
        model.summary()
        model.build(input_shape=(None, MAX_LEN))

        return model