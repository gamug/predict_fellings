import os, re, sys, warnings, psutil, nltk, string, unicodedata, time, datetime
from .database import socialDb
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Flatten, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

class fellingsAnalysis():
    def __init__(self, dataset, name='default'):
        '''''
        Object to create and train model
            Input:
                dataset: socialDb object
                name: Name to save trained model   
        '''''
        assert type(dataset) == socialDb, f'database must be socialDb find {type(dataset)}'
        self.workPath, self.dataset, self.name = dataset.workPath, dataset, name
        self.getSets()
        print('    building model')
        self.NN = self.buildModel()

    def getSets(self, y_col=['Negative', 'Neutral', 'Positive']):
        '''''
        split dataset into x_train, y_train, x_test and y_test saving it as attributes
        '''''
        self.x_train = np.array(list(self.dataset.train['indexs']), dtype=np.int64)
        self.y_train = np.array(self.dataset.train[y_col], dtype=np.int64)
        self.x_test = np.array(list(self.dataset.test['indexs']), dtype=np.int64)
        self.y_test = np.array(self.dataset.test[y_col], dtype=np.int64)

    def buildModel(self):
        '''''
        asigne model architecture. change this method if u' wanna change NN
        '''''
        sentence_indices = Input((self.dataset.sent_maxlen,), name='input_')
        embeddings = self.dataset.embed.get_keras_embedding()(sentence_indices)
        #         X = Bidirectional(LSTM(units=128, return_sequences=True, name='lstm_1'))(embeddings)
        X = LSTM(units=128, return_sequences=True, name='lstm_1')(embeddings)
        X = Dropout(rate=0.4, name='drop_1')(X)
        X = LSTM(units=128, return_sequences=True, name='lstm_2')(X)
        X = Dropout(rate=0.4, name='drop_2')(X)
        X = Flatten()(X)
        X = Dense(len(self.y_test[0]), name='dense_', activation='sigmoid')(X)
        model = Model(inputs=sentence_indices, outputs=X)
        return model

    def trainModel(self, epochs, loss='binary_crossentropy', batch=32):
        '''''
        train model with dataset got in getDataset
        '''''
        print('    training model')
        self.NN.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        self.NN.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch, shuffle=True)
        return

    def saveModel(self):
        '''''
        save model using name
        '''''
        print('    saving model')
        if self.name == 'default':
            now = datetime.datetime.now().strftime("%d-%m-%Y")
            self.NN.save(os.path.join(self.workPath, 'model', f'model_{now}.h5'))
        else:
            self.NN.save(os.path.join(self.workPath, 'model', f'model_{self.name}.h5'))