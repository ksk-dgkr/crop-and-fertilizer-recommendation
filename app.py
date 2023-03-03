import numpy as np
from flask import Flask,request,render_template
import pickle
import joblib
import json

model_crop=pickle.load(open('crop_mod.pkl','rb'))
model_fert=joblib.load(open('knn_fert_mod_inpNum.pkl','rb'))

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
        SoilType = {'Sandy':4, 'Loamy':2, 'Black':0, 'Red':3, 'Clayey':1}
        CropType = {"Sugarcane": 8,"Cotton": 1,"Millets": 4,"Paddy": 6,"Pulses":7,
                    "Wheat": 10,"Tobacco": 9,"Barley": 0,"Oil seeds": 5,
                    "Ground Nuts": 2,"Maize": 3}
        FertType = {0 : "Urea",1 : "DAP",2 : "28-28",3 : "14-35-14",4 : "20-20",
            5 : "17-17-17",6 : "10-26-26"}
        data1=request.form['Temperature']
        data2=request.form['Humidity']
        data3=request.form['Moisture']
        soil=request.form['Soil_Type']
        data4=SoilType[soil]
        crop=request.form['Crop_Type']
        data5=CropType[crop]
        data6=request.form['Nitrogen'] 
        data7=request.form['Potassium']
        data8=request.form['Phosphorous']
        arr1=np.array([[data1,data2,data3,data4,data5,data6,data7,data8]], dtype=float)
        print(arr1)
        fert_pred = model_fert.predict(arr1)[0]
        givenVal = [[data1,data2,data3,soil,crop,data6,data7,data8]]
        return render_template('result.html', data = fert_pred, givenValues = givenVal)
    else:
        return render_template('fertilizer.html')
    
if __name__=="__main__" :
    app.run(debug=True,port=8000, use_reloader = False)