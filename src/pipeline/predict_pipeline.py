import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scale = preprocessor.transform(features)
            
            pred  = model.predict(data_scale)
            return pred 
        
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(
        self,
        JobTitle: str,
        country: str,
        gender: str,
        age: int,
        BasePay: float,
        OvertimePay: float,
        OtherPay: float,
        TotalPay: float,
        TotalPayBenefits: float,
        CreditCardDebt: float,
        NetWorth: float
    ):
        self.JobTitle = JobTitle
        self.country = country
        self.gender = gender
        self.age = age
        self.BasePay = BasePay
        self.OvertimePay = OvertimePay
        self.OtherPay = OtherPay
        self.TotalPay = TotalPay
        self.TotalPayBenefits = TotalPayBenefits
        self.CreditCardDebt = CreditCardDebt
        self.NetWorth = NetWorth
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'JobTitle': [self.JobTitle],
                'country': [self.country],
                'gender': [self.gender],
                'age': [self.age],
                'BasePay': [self.BasePay],
                'OvertimePay': [self.OvertimePay],
                'OtherPay': [self.OtherPay],
                'TotalPay': [self.TotalPay],
                'TotalPayBenefits': [self.TotalPayBenefits],
                'credit card debt': [self.CreditCardDebt],
                'net worth': [self.NetWorth]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
