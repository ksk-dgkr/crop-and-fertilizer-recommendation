import numpy as np
from flask import Flask,request,render_template
import pickle

model=pickle.load(open('mod.pkl','rb'))

app=Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data1=request.form('Nitrogen') 
    data2=request.form('Phosphorous') 
    data3=request.form('Potassium')
    data4=request.form('temperature')
    data5=request.form('ph') 
    data6=request.form('humidity') 
    data7=request.form('rainfall') 
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7]])
    pred=model.predict(arr)
    return render_template('after.html',data=pred)

if __name__=="__main__" :
    app.run(debug=True,port=8000)
