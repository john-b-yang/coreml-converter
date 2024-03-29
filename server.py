import os, shutil
from flask import Flask, render_template, request, flash, url_for, redirect
from mlfunction.script import conversion

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'testkey'

# Functionalities
def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

# Routing
@app.route('/')
def index():
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
    params = list(request.form.values())

    if 'csv' in request.files:
        # print(list(request.files))
        file = request.files['csv']
        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(f)
        print("File Name: {0}, Path: {1}".format(file.filename, f))
    else:
        flash("Test Message 1")

    print(params)
    split = request.form['split']
    y_col = request.form['label']
    print("Split: {0}. Y-Col: {1}".format(split, y_col))

    returnIndex = False

    intersects = bool(set(['svc', 'knn', 'decision-trees', 'random-forest', 'gradient-boosted']).intersection(params))
    if not intersects:
        returnIndex = True
        flash('Please select at least one model to test')

    if not split:
        returnIndex = True
        flash('Please remember to enter a split value')

    if not y_col:
        returnIndex = True
        flash('Please enter the name of the column containing label data')

    if returnIndex:
        return redirect(url_for('index'))
    else:
        str_params = " ".join(params)
        result = conversion(f, split, str_params, y_col)
        print("Results: %s" % result)

        # Assuming output.txt is created, duplicate the file and reconfigure for display
        duplicate = open('output2.txt', 'w')
        shutil.copyfile('output.txt', 'output2.txt')
        line_prepender('output2.txt', '<link href="txtstyle.css" rel="stylesheet" type="text/css" />')

        # Converting txt file into html file and moving it into templates directory
        os.rename('output2.txt', 'output.html')
        shutil.copy('output.html', 'templates')
        os.remove('output.html')

        return render_template('results.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)
