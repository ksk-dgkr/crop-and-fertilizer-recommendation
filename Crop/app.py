import numpy as np
from flask import Flask,request,render_template
import pickle

model=pickle.load(open('mod.pkl','rb'))

app=Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data1=request.form['N']
        data2=request.form['P']
        data3=request.form['K']
        data4=request.form['temperature']
        data5=request.form['humidity']
        data6=request.form['ph'] 
        data7=request.form['rainfall']
        arr=np.array([[data1,data2,data3,data4,data5,data6,data7]])
        print(arr)
        pred = model.predict(arr)[0]
        return render_template('after.html', data = pred, givenValues = arr)
    else:
        return render_template('index.html')

if __name__=="__main__" :
    app.run(debug=True,port=8000)
