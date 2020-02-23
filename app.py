import os 
import shutil
from flask import Flask, render_template, request


from main import *

UPLOAD_FOLDER = "static/images"

for the_file in os.listdir(UPLOAD_FOLDER):
    file_path = os.path.join(UPLOAD_FOLDER, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        
    except Exception as e:
        print(e)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':

        file = request.files['file']
        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(f)
        name = file.filename
        image = UPLOAD_FOLDER + '/' + name

        found, flag = breed(path = image)
        if flag == 0:
            info = wiki(found)
            top = 'More about ' + found
        else:
            info = ''
            top = 'NO DOG FOUND  | UPLOAD A DOG IMAGE.'
            
        
        if device == 'cpu':
            chip = 'CPU'
        else:
            chip = 'GPU | available'
        
        return render_template('result.html',breed_name=found,  info= info, device= chip, image_name = name, head = top)
        


if __name__ == '__main__':
    app.run(debug =True)


