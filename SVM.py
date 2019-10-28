from sklearn import svm

import json 

class SVM :

    def __init__(self):
        self.SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        self.encodeDict = {
                    0 : "negative",
                    1 : "positive"
                }

    # vectorized train_x by TF-IDF
    def fit(self, train_x, train_y):
        self.SVM.fit(train_x, train_y)
    
    # vectorized test_x by TF-IDF
    def predict(self, test_x):
        return self.SVM.predict(test_x) 
    
    def mergeJSON(self, result_vector, dataset_json): 
        positiveNegativeList = list(map(self.encodeDict.get, result_vector))
        output = {}
        dataset_json = json.loads(dataset_json)
        output['text'] = dataset_json['text']
        output['label'] = dataset_json['label']
        output['label_after'] = {}

        index = 0 
        for i in positiveNegativeList : 
            output['label_after'][index] = positiveNegativeList[index] 
            index = index + 1

        return output
