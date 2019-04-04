from flask import Flask, render_template, request
app = Flask(__name__)

from main import *

@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        file = request.files['file']
        image = file.read()
        pas = breed(path = image)
        gpu = device
        
        return render_template('result.html',breed_name=pas, processor = gpu)
        


if __name__ == '__main__':
    app.run(debug =True)
