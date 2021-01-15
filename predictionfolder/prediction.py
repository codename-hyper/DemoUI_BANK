from preprocessingfolder import preprocessingfile
import pandas as pd
import numpy as np
import joblib
import pickle
from Data_ingestion import data_ingestion

class predict():
    def __init__(self):
        pass

    def predictor(self,file):
        instance1 = data_ingestion.data_getter()
        data = instance1.data_load(file)
        df = data[['type','nameOrig','nameDest']].copy(deep  =True)
        instance2 = preprocessingfile.preprocess()

        set1=instance2.new_feature(data)
        set2=instance2.drop_columns(set1)

        model = joblib.load('pickle_files/pickle_fraud.pkl')
        ss = joblib.load('pickle_files/scaler_fraud.pkl')
        en = joblib.load('pickle_files/encode_fraud.pkl')
        set2[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig',
              'errorBalanceDest']] = ss.transform(set2[['amount', 'oldbalanceOrg', 'newbalanceOrig',
                                                            'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig',
                                                            'errorBalanceDest']])
        set2.type = en.transform(set2.type)
        result = model.predict(set2)

        df['output'] = result

        df['output']=np.where(df['output']==0,"Trusted","Fraud")
        return df
# ===================================================================================================
class Fraud_predict():
    def __init__(self):
        pass

    def predictor(self,file):
        instance1 = data_ingestion.data_getter()
        data = instance1.data_load(file)

        instance2 = preprocessingfile.Fraud_preprocess()

        set0 = instance2.initialize_columns(data)
        set1 = instance2.drop_columns(set0)
        df = set1[['customer']]
        set2 = set1.drop(['customer'],axis=1)
        new_data = instance2.obj_to_cat(set2)

        model_Fraud = joblib.load('pickle_files/Fraud_new_model.pkl')
        result = model_Fraud.predict(new_data)

        df['output'] = result
        df['output'] = np.where(df['output'] == 0, "Trusted", "Fraud")
        return df

# ===================================================================================================
class LA_predict():
    def __init__(self):
        pass

    def predictor(self,file):
        instance1 = data_ingestion.data_getter()
        data = instance1.data_load(file)

        instance2 = preprocessingfile.LA_preprocess()

        set0 = instance2.initialize_columns(data)
        set1 = instance2.imputer(set0)
        final_data = set1[['Loan_ID']]
        set2 = instance2.drop_Loan_id(set1)
        encoded_data = instance2.encode_cat_f(set2)
        new_data = instance2.drop_columns(encoded_data)

        ss_LA = pickle.load(open('pickle_files/veena_LA_stan_scaler.pkl', 'rb'))
        model_LA=pickle.load(open('pickle_files/LAOpt8model.sav', 'rb'))

        ss_result = ss_LA.fit_transform(new_data)
        result = model_LA.predict(ss_result)

        new_data['output'] = result
        new_data['output'] = np.where(new_data['output'] == 0,"Rejected","Accepted")
        final_data['Output']=new_data['output']
        return final_data







