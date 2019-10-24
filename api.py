import os
from flask import Flask, render_template, url_for, json

app = Flask(__name__) 

@app.route('/')
def hello_world():
    return 'Hello, world!' 

@app.route('/result') 
def showjson():
    filename = os.path.join(app.static_folder, 'result.json')
    with open(filename) as blog_file:
        data = json.load(blog_file)
        return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

