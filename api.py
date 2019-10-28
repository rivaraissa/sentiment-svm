import os
from flask import Flask, render_template, url_for, json
from flask_cors import CORS 

app = Flask(__name__) 
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, world!' 

@app.route('/diagram') 
def showjson():
    filename = os.path.join(app.static_folder, 'diagram.json')
    with open(filename) as blog_file:
        data = json.load(blog_file)
        return data

@app.route('/result') 
def showResultJson():
    filename = os.path.join(app.static_folder, 'result.json')
    with open(filename) as blog_file:
        data = json.load(blog_file)
        return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

