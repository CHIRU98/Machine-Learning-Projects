import numpy as np
import pickle
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('adaboost.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    gender = request.form['gender']
    heart_disease = request.form['heart_disease']
    ever_married = request.form['ever_married']
    work_type = request.form['work_type']
    Residence_type = request.form['Residence_type']
    smoking_status = request.form['smoking_status']

    Hypertension = request.form.get("Hypertension")
    avg_glucose_level = request.form.get("avg_glucose_level")
    bmi = request.form.get("bmi")
    age = request.form.get("Age")
    

    columns = ['avg_glucose_level','bmi','age']

    data = pd.DataFrame([[gender,heart_disease,ever_married,work_type,Residence_type,smoking_status,Hypertension,avg_glucose_level,bmi,age]],
                                       columns=['gender','heart_disease','ever_married','work_type','Residence_type','smoking_status','Hypertension','avg_glucose_level','bmi','age'],
                                       )
    data[columns] = sc.fit_transform(data[['avg_glucose_level','bmi','age']])
    
    prediction = model.predict(data)[0]

    if int(prediction)== 1: 
            prediction ='You Have Stroke'
    else: 
        prediction ='Congrats,You Dont have any disease'

    return render_template("result.html",data=prediction)

if __name__ == "__main__":
    app.run()
# if __name__ == "__main__":
#     app.run(host='0.0.0.0',port=8080)
