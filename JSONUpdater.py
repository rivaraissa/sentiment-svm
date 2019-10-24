import json 

class JSONUpdater : 
    def __init__(self): 
        self.data = {}
        self.file_path = "static/result.json"
        self.data['real'] = {}
        self.data['training'] = {}
        self.data['test'] = {}
        self.data['test_after'] = {}

    def set_percentage(
            self,
            real_pos,
            real_neg,
            training_pos,
            training_neg,
            test_pos,
            test_neg,
            test_after_pos,
            test_after_neg):

        self.data['real']['pos'] = real_pos
        self.data['real']['neg'] = real_neg
        self.data['training']['pos'] = training_pos
        self.data['training']['neg'] = training_neg
        self.data['test']['pos'] = test_pos
        self.data['test']['neg'] = test_neg
        self.data['test_after']['pos'] = test_after_pos
        self.data['test_after']['neg'] = test_after_neg 

        self.write()

    def write(self):
        with open(self.file_path, 'w') as outfile: 
            json.dump(self.data, outfile) 

