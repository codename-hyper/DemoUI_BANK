import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

class preprocess:

    def initialize_columns(self, data):
        data = data
        try:
            data.columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
                                   'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
            return data

        except Exception as e:
            raise e

    def new_feature(self, data):
        try:
            data['errorBalanceOrig'] = data["amount"] + data["newbalanceOrig"] - data["oldbalanceOrg"]
            data['errorBalanceDest'] = data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]
            return data
        except Exception as e:
            raise e

    def drop_columns(self,data):
        try:
            data.drop(['step', 'nameOrig', 'isFlaggedFraud', 'nameDest'], inplace=True, axis=1)
            return data
        except Exception as e:
            raise e

#=======================================================================================
class Fraud_preprocess:
    def initialize_columns(self, data):  # depend ......................
        data.columns = ['step', 'customer', 'age', 'gender', 'zipcodeOri', 'merchant',
                        'zipMerchant', 'category', 'amount']
        return data

    def drop_columns(self, data):
        data_reduced = data.drop(['zipcodeOri', 'zipMerchant'], axis=1)
        return data_reduced

    def obj_to_cat(self, data_reduced):
        col_categorical = data_reduced.select_dtypes(include=['object']).columns
        for col in col_categorical:
            data_reduced[col] = data_reduced[col].astype('category')
        # categorical values ==> numeric values
        data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)
        return data_reduced

# ======================================================================================================================
class LA_preprocess:
    def initialize_columns(self, data):  # depend ......................
        data.columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                        'Credit_History', 'Property_Area']
        return data

    def imputer(self, data):
        df_num = data.select_dtypes(include=np.number)
        df_cat = data.select_dtypes(include=object)
        # for numerical
        imp = SimpleImputer(strategy='mean')
        imp_num = imp.fit(df_num)
        num_test = imp_num.transform(df_num)
        test_num = pd.DataFrame(num_test)
        test_num.columns = df_num.columns
        # for categorical
        imp = SimpleImputer(strategy='most_frequent')
        imp_cat = imp.fit(df_cat)
        cat_test = imp_cat.transform(df_cat)
        test_cat = pd.DataFrame(cat_test)
        test_cat.columns = df_cat.columns
        # concatinate
        frames = [test_cat, test_num]
        test_df = pd.concat(frames, axis=1)
        return test_df

    def drop_Loan_id(self, data):  # str loan ............
        new_data = data.drop(['Loan_ID'], axis=1)
        return new_data

    def encode_cat_f(self, data):
        data_encoded = pd.get_dummies(data, drop_first=True)
        return data_encoded

    def drop_columns(self, data):
        new_data = data.drop(["Gender_Male", "Dependents_1", "Dependents_2", "Dependents_3+",
                              "Education_Not Graduate", "Self_Employed_Yes"], axis=1)
        return new_data



