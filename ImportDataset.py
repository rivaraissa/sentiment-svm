import pandas as pd 
import json 

class ImportDataset : 

    def __init__(self):
        self.corpus = ''
        self.text = ''
        self.label = ''

    def readCsv(self, csvUrl):
        self.corpus = pd.read_csv('komentar.csv', encoding='latin-1') 
        self.text = self.corpus['text']
        self.label = self.corpus['label'] 

    def getCorpus(self):
        return self.corpus
    
    def getText(self):
        return self.text

    def getLabel(self): 
        return self.label

    def toJSON(self): 
        return self.corpus.to_json()
