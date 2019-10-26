import json 

class JSONUpdater : 
    def __init__(self): 
        self.data = {}
        self.file_path = "static/diagram.json"
        self.all_result = "static/result.json"
        self.data['real'] = {}
        self.data['training'] = {}
        self.data['test'] = {}
        self.data['test_after'] = {}
        self.data['all_data'] = {}

    def set_percentage(
            self,
            real_pos,
            real_neg,
            training_pos,
            training_neg,
            test_pos,
            test_neg,
            test_after_pos,
            test_after_neg,
            all_svm_pos,
            all_svm_neg):

        self.data['real']['pos'] = real_pos
        self.data['real']['neg'] = real_neg
        self.data['training']['pos'] = training_pos
        self.data['training']['neg'] = training_neg
        self.data['test']['pos'] = test_pos
        self.data['test']['neg'] = test_neg
        self.data['test_after']['pos'] = test_after_pos
        self.data['test_after']['neg'] = test_after_neg 
        self.data['all_data']['pos'] = all_svm_pos 
        self.data['all_data']['neg'] = all_svm_neg

        self.write_diagram()

    def write_diagram(self):
        with open(self.file_path, 'w') as outfile: 
            json.dump(self.data, outfile) 
    
    def write_result(self, json_file):
        with open(self.all_result, 'w') as outfile:
            json.dump(json_file, outfile)
