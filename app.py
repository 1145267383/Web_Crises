from flask import Flask,render_template,request
import joblib
import numpy as np
from helpers.dummies import *

app=Flask(__name__)

model_s=joblib.load('models/model_s.h5')
model_b=joblib.load('models/model_b.h5')
scaler=joblib.load('models/scaler.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/hast')
def hast():
    return render_template('hastory.html')

@app.route('/predict',methods=['GET'])
def predict():
    all_data=request.args

    x1_year=int(all_data["year"])
    x2_gstd=int(all_data["gstd"])
    x3_exus=float(all_data["exus"])
    x4_domdeb=int(all_data["domdeb"])
    x5_sovext1=int(all_data["sovext1"])
    x6_sovext2=int(all_data["sovext2"])
    x7_gdp=float(all_data["gdp"])
    x8_inf=float(all_data["inf"])
    x9_ind=int(all_data["ind"])
    x10_curcs=int(all_data["curcs"])
    x11_infcs=int(all_data["infcs"])
    x12_lavel=lavel_dummies[all_data['lavel']]

    data=[x1_year,x2_gstd,x3_exus,x4_domdeb,x5_sovext1,x6_sovext2,x7_gdp,x8_inf,x9_ind,x10_curcs,x11_infcs]+x12_lavel

    d=np.array(data)
    ds=scaler.transform([d[[0,2,6,7]]])
    data_scaled=np.array([[ds[0][0],d[1],ds[0][1],d[3],d[4],d[5],ds[0][2],ds[0][3],d[8],d[9],d[10],d[11],d[12]]])
    
    pred_s=model_s.predict(data_scaled)[0]
    pred_b=model_b.predict(data_scaled)[0]

    return render_template('prediction.html',Sys_crises=predict_dummies[pred_s] ,Bnk_crises=predict_dummies[pred_b] )


if __name__=='__main__':
    app.run(debug=True, port=9000)