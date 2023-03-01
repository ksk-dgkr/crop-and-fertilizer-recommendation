import numpy as np
from flask import Flask,request,render_template
import pickle
import joblib

model_crop=pickle.load(open('crop_mod.pkl','rb'))
model_fert=joblib.load(open('fert_mod.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/crop_inp', methods = ['POST', 'GET'])
def crop_inp():
    return render_template('crop.html')

@app.route('/fert_inp', methods = ['POST', 'GET'])
def fert_inp():
    return render_template('fertilizer.html')

@app.route('/predict_crop', methods=['POST', 'GET'])
def predict_crop():
    if request.method == 'POST':
        data1=request.form['N']
        data2=request.form['P']
        data3=request.form['K']
        data4=request.form['temperature']
        data5=request.form['humidity']
        data6=request.form['ph'] 
        data7=request.form['rainfall']
        arr=np.array([[data1,data2,data3,data4,data5,data6,data7]])
        #print(arr)
        pred = model_crop.predict(arr)[0]
        return render_template('after.html', data = pred, givenValues = arr)
    else:
        return render_template('crop.html')
    

@app.route('/predict_fert', methods=['POST', 'GET'])
def predict_fert():
    if request.method == 'POST':
        data1=request.form['Temperature']
        data2=request.form['Humidity']
        data3=request.form['Moisture']
        data4=request.form['Soil_Type']
        data5=request.form['Crop_Type']
        data6=request.form['Nitrogen'] 
        data7=request.form['Potassium']
        data8=request.form['Phosphorous']
        arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8]])
        #print(arr)
        pred = model_fert.predict(arr)
        return render_template('result.html', data = pred, givenValues = arr)
    else:
        return render_template('fertilizer.html')
    
if __name__=="__main__" :
    app.run(debug=True,port=8000, use_reloader = False)