import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import CustomLogger
from src.utils import save_object

logger =CustomLogger()

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifact','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        """
        This function is responsible for data transformation
        
        Authore: rahul patil
        email:rrp30998@gmail.com
        
        """
        try:
            numerical_columns = ['age', 'BasePay', 'OvertimePay', 'OtherPay', 'TotalPay',
        'TotalPayBenefits', 'credit card debt', 'net worth']
            categorical_columns  = ['JobTitle', 'country']
            
            num_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ],
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('OrdinalEncoder',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)),
                ]
            )
            
            logger.log('Numerical columns standard scaling completed')
            logger.log('categorical columns standard scaling completed')
            
            preprocess = ColumnTransformer(
                [
                    ('num_pipelene',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ],
                remainder='passthrough'
            )
            
            return preprocess
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transofrmation(self,train_path,test_path):
        """
        this functional actually perform data trnasormation using object of 'get_data_transformation_object' function 
        
        """
        try:
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)
            df = pd.concat([df_train, df_test], axis=0)
            logger.log('redaing train and test data completed')
            logger.log('obtaining preprocess object')
            
            #create object of  data transformation pipeline
            preprocessing_obj= self.get_data_transformation_object()
            
            target_column ='car purchase amount'
            
            input_feature_df = df.drop(columns=[target_column],axis=1)
            target_feature_df = df[target_column]
            
            logger.log('appying preprocessing object on training and testing dataframe')
            
            input_feature_arr = preprocessing_obj.fit_transform(input_feature_df)

            input_feature_train_arr,input_feature_test_arr,target_feature_train_df,target_feature_test_df=train_test_split(input_feature_arr, target_feature_df, test_size=0.2, random_state=42)
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logger.log('save preprocessing onject')
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessing_obj
            )
            return train_arr,test_arr
        
        except Exception as e:
            raise CustomException(e,sys)
            
            