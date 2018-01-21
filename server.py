import os
from flask import Flask, render_template, request
from mlfunction.script import conversion

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['csv']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)
    print(file.filename)
    print(f)
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_results():
    print(list(request.form.values()))
    print(type(request.form))
    split = request.form['split']
    y_col = request.form['label']
    print("Split: {0}. Y-Col: {1}".format(split, y_col))

    file = request.files['csv']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)
    print("Path: {0}, File Name: {1}".format(file.filename, f))

    return render_template('results.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)
