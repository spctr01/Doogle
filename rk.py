from flask import Flask, render_template, request
app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        r = "german shephard"
        file = request.files['file']
        image = file.read()
   
        
        return render_template('result.html',breed_name=r)
        


if __name__ == '__main__':
    app.run(debug =True)
