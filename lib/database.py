import os, re, sys, warnings, psutil, nltk, string, unicodedata, time, datetime
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


class socialDb():
    def __init__(self, workPath, name='default', sent_maxlen=200, test_size=0.4):
        '''''
        class designed to process social media comments database
            input:
                workPath: workPath where is located folder structure
                name: name to save database
                sent_maxlen: maximun lenght of sentence to pad-cut sentence
                test_size: test dataset portion of dataset.
        '''''
        self.workPath, self.sent_maxlen, self.name, self.test_size = workPath, sent_maxlen, name, test_size
        self.stop_words = stopwords.words('spanish')
        self.stop_words.remove('no')
        self.dataset = pd.DataFrame()
        print(r'    charging embeding model')
        self.embed = KeyedVectors.load(os.path.join(self.workPath, r"wordEmbedding/complete.kv"), mmap='r')

    def getDatabase(self, database):
        '''''
        Charge database from "dataset" folder. This function let charge .csv, xlsx, folders with
        multiple csv, xlsx files and pre-processed databases.
            input:
                database: file or folder name
            return:
                db: database charged
        '''''
        assert type(database) == str, 'database type error: database must be "str" type'
        ext, names = [], []
        for file in os.listdir(os.path.join(self.workPath, r'dataset')):
            names.append(file.split('.')[0])
            try:
                ext.append(file.split('.')[1])
            except:
                ext.append('')
        ext, names = np.array(ext, dtype=str), np.array(names, dtype=str)
        assert database in names, f'database not found in path {os.path.join(self.workPath, "database", database)} .csv,'
        ' .xlsx or folder'
        if re.match('xl.*', ext[np.where(names == database)[0][0]]):
            db = pd.read_excel(
                os.path.join(self.workPath, r'dataset', f'{database}.{ext[np.where(names == database)[0][0]]}')
            )
        elif ext[np.where(names == database)[0][0]] == 'csv':
            db = pd.read_csv(
                os.path.join(self.workPath, r'dataset', f'{database}.{ext[np.where(names == database)[0][0]]}')
            )
        elif ext[np.where(names == database)[0][0]] == '':
            db = pd.DataFrame()
            print(r'    processing folder database')
            for file in os.listdir(os.path.join(self.workPath, 'dataset', database)):
                print(f'      processing {file}')
                try:
                    appendable = pd.read_csv(os.path.join(self.workPath, 'dataset', database, file)).dropna()
                except:
                    appendable = pd.read_excel(os.path.join(self.workPath, 'dataset', database, file)).dropna()
                appendable = appendable[appendable['CONTENT'] != 'Content not available']
                db = db.append(appendable, ignore_index=True)
                db.drop_duplicates(subset='CONTENT', inplace=True)
            db.reset_index(inplace=True)
        if list(db.columns) == ['Unnamed: 0', 'pros_text', 'indexs', 'factorize']:
            db = pd.read_csv(os.path.join(self.workPath, 'dataset', 'dataset_down_group.csv'))
            db['indexs'] = [np.fromstring(
                re.sub('\]', '', re.sub(r'\[', '', db.indexs.iloc[i])), sep=',', dtype=np.int
            ).reshape((self.sent_maxlen,)) for i in range(len(db))]
            db.drop('Unnamed: 0', axis=1, inplace=True)
            self.dataset = db
            return
        self.dataset = pd.DataFrame()
        db.drop_duplicates(subset='CONTENT', inplace=True)
        db.to_excel(os.path.join(self.workPath, 'database_tocateg.xlsx'))
        return db

    def processText(self, database, column):
        '''''
        Process database text to preparing dataset. Database must be a folder or file located in "dataset"
        folder
            input:
                database: file or folder name to charge as database
                column: text column from database
        '''''
        assert type(database) == str, 'columType error: must be string'
        db = self.getDatabase(database)
        if len(self.dataset) > 0:
            print('    you charged a pre-processed database. it is ready to use in train-predict')
            return
        assert type(column) == str, 'columType error: must be string'
        assert column in db.columns, f'column not in db columns: {db.columns}'
        zero = self.embed.vocab['httpwwwfacebookcomellasdemontena'].index
        url_rex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]' \
                  r'[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.' \
                  r'[a-zA-Z0-9]+\.[^\s]{2,})+[^\s]+'
        texts, pros, indexs = db[column], [], []
        print(r'    processing texts')
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f'      sentence {i} from {len(texts)}')
            text = re.sub(r'^[^\s]+: ', '', text)
            text = re.sub(url_rex, '', text)
            text = re.sub(r'@.[^\s]+', '', text)
            text = re.sub(r'#.[^\s]+', '', text)
            text = re.sub(r'[^a-zA-Z\d\sáéíóúñü]', '', text)
            text = re.sub(r' [A-Za-z] ', ' ', text)
            text = re.sub(r' {1,}', ' ', text).lower()
            text = ' '.join([word for word in text.split(' ') if not word in self.stop_words])
            index = []
            for i in range(self.sent_maxlen):
                try:
                    index.append(self.embed.vocab[text.split(' ')[i]].index)
                except:
                    index.append(zero)
            indexs.append(np.array(index, dtype=np.int64).reshape((self.sent_maxlen,)))
            pros.append(text)
        self.dataset = pd.get_dummies(db['SENTIMENT'])
        self.dataset.insert(0, 'indexs', indexs)
        self.dataset.insert(0, 'pros_text', pros)
        self.train, self.test = train_test_split(self.dataset, test_size=self.test_size, random_state=12345)

    def saveDatabase(self):
        print('    saving datasets')
        assert len(self.dataset) > 0, "there isn't dataset to save, run processText method"
        if self.name == 'default':
            now = datetime.datetime.now().strftime("%d-%m-%Y")
            print('        saving total dataset...')
            self.dataset.to_csv(os.path.join(self.workPath, 'dataset', f'dataset_{now}.csv'))
            print('        saving train set...')
            self.train.to_csv(os.path.join(self.workPath, 'dataset', f'train_{now}.csv'))
            print('        saving test set...')
            self.test.to_csv(os.path.join(self.workPath, 'dataset', f'test_{now}.csv'))
        else:
            print('        saving total dataset...')
            self.dataset.to_csv(os.path.join(self.workPath, 'dataset', f'dataset_{self.name}.csv'))
            print('        saving train set...')
            self.train.to_csv(os.path.join(self.workPath, 'dataset', f'train_{self.name}.csv'))
            print('        saving test set...')
            self.test.to_csv(os.path.join(self.workPath, 'dataset', f'test_{self.name}.csv'))