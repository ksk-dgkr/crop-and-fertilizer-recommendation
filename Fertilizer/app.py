import numpy as np
from flask import Flask,request,render_template
import joblib

model=joblib.load(open('mod.pkl','rb'))

app=Flask(__name__)



@app.route('/')
def home():
    return render_template('fertilizer.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
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
        print(arr)
        pred = model.predict(arr)
        return render_template('result.html', data = pred, givenValues = arr)
    else:
        return render_template('fertilizer.html')

if __name__=="__main__" :
    app.run(debug=True,port=8000)
