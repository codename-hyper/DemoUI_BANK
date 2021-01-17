from flask import Flask, render_template,url_for, request,redirect
import pandas as pd
import pickle
import warnings
import joblib
from flask_cors import CORS, cross_origin
from predictionfolder.prediction import LA_predict,predict,Fraud_predict,LR_predict

def warns(*args, **kwargs):
    pass
warnings.warn = warns

ALLOWED_EXTENSIONS = set(['csv','xlsx','data'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# load the model from directory
#model = joblib.load('pickle_files/pickle_fraud.pkl')
#ss = joblib.load('pickle_files/scaler_fraud.pkl')
ss_LA = pickle.load(open('pickle_files/veena_LA_stan_scaler.pkl', 'rb'))
model_LA=pickle.load(open('pickle_files/LAOpt8model.sav', 'rb'))
model_Fraud = joblib.load('pickle_files/Fraud_new_model.pkl')

instance = predict()
LA_instance = LA_predict()
Fraud_instance = Fraud_predict()
LR_instance = LR_predict()
app = Flask(__name__)

@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/LA',methods=['GET','POST'])
@cross_origin()
def LA():
    if request.method == 'POST':
        return render_template('neha_secondsample.html')

@app.route('/FD',methods=['GET','POST'])
@cross_origin()
def FD():
    if request.method == 'POST':
        return render_template('FD_new_home.html')

@app.route('/LR',methods=['GET','POST'])
@cross_origin()
def LR():
    if request.method == 'POST':
        return render_template('loan_risk.html')

@app.route('/bulk_predict',methods=['GET','POST'])
@cross_origin()
def bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            data = Fraud_instance.predictor(file)
            return render_template('result_bulk.html', tables=[data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)

@app.route('/LA_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LA_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            data = LA_instance.predictor(file)
            return render_template('result_bulk.html', tables=[data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)

@app.route('/LR_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LR_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            data = LR_instance.predictor(file)
            return render_template('result_bulk.html', tables=[data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method == 'POST':

        Type = request.form.get("gender", False)
        if (Type == 'Male'):
            Type = 0
        elif (Type == 'Female'):
            Type = 1
        elif (Type == 'Enterprise'):  # random
            Type = 2
        elif (Type == 'Unknown'):
            Type = 3

        amount = float(request.form.get("amount", False))
        merchant = float(request.form.get("merchant", False))
        category = float(request.form.get("category", False))
        step = float(request.form.get("step", False))
        age = float(request.form.get("age", False))

        df = pd.DataFrame(
            {"step": step ,"age": age,'gender': Type,  "merchant": merchant,
                    "category": category,"amount": amount  }, index=[0])

        my_prediction = model_Fraud.predict(df)

        # check my_pred ..............................................
        return render_template('fraud_detect_result.html', prediction=my_prediction)

@app.route('/predict_LA',methods=['GET','POST'])
@cross_origin()
def predict_LA():
    if request.method == 'POST':
        married = request.form.get("Married", False)
        if (married=='Yes'):
            married=1
        elif(married=="No"):
            married=0

        property = request.form.get("Property", False)
        if (property == 'Semi-Urban'):
            property_semi = 1
        else:
            property_semi = 0
        if (property == "Urban"):
            property_urban = 1
        else:
            property_urban = 0

        ApplicantIncome = float(request.form.get("ApplicantIncome",False))
        CoapplicantIncome = float(request.form.get("CoapplicantIncome",False))
        LoanAmount = float(request.form.get("LoanAmount",False))
        Loan_Amount_Term = float(request.form.get("Loan_Amount_Term",False))
        Credit_History = float(request.form.get("Credit_History",False))

        LA_prediction = model_LA.predict(ss_LA.fit_transform([[ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History
                                                         ,married,property_semi,property_urban]]))

        return render_template('LA_result.html', prediction=LA_prediction)
    else:
        return render_template('LA_home.html')

if __name__ == '__main__':
    # To run on web ..
    #app.run(host='0.0.0.0',port=8080)
    # To run locally ..
    app.run(debug=True)