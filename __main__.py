import os, warnings
import pandas as pd
from lib.database import socialDb
from lib.model import fellingsAnalysis
warnings.filterwarnings("ignore")
workPath = r'C:\Users\g.munera.gonzalez\Desktop\Fellings prediction'
print('processing database')
db = socialDb(workPath, name='down_group')
db.processText('DESCARGAS GRUPALES', 'CONTENT')
#db.saveDatabase()
# print('creating and training model')
# model = fellingsAnalysis(db, 'covid')
# model.trainModel(6)
# model.saveModel()
# acc = model.NN.evaluate(model.x_test, model.y_test)
# print(f'loss: {acc[0]}, acuraccy: {acc[1]}')
# aux = model.NN.predict(model.x_test)
# pd.DataFrame(data=aux, columns=['Negative', 'Neutral', 'Positive']).to_excel(
#     os.path.join(workPath, 'dataset', 'pedict_test.xlsx')
# )