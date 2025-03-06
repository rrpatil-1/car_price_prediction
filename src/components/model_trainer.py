import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.linear_model import LinearRegression,Lasso, Ridge
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import CustomLogger
from src.utils import evaluate_model, save_object
from dataclasses import dataclass


logger = CustomLogger()

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifact','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_arry,test_arry):
        try:
            logger.log('spliting tarining and test input data')
            X_train,Y_train,X_test,Y_test=(
                train_arry[:,:-1],
                train_arry[:,-1],
                test_arry[:,:-1],
                test_arry[:,-1]
                )
            models ={
                    "Linear Regression":LinearRegression(),
                    "Lasso":Lasso(alpha=100,max_iter=1000,selection='random'),
                    "Ridge":Ridge(alpha=1,solver='sag')}
            param ={
                'Linear Regression':{

                },
                'Lasso':{
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    'max_iter': [1000, 2000],
                    'selection': ['cyclic', 'random']
                },
                'Ridge':{
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
                         }
            }
            
            model_report:dict=evaluate_model(X_train,Y_train,X_test,Y_test,models,param)
            print(model_report)
            #To get best model score
            best_model_score = max(model_report.values())
            print(best_model_score)
            #To get best model name from  dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException('No best model found',sys)
            
            logger.log('best model found on training and testing dataset')
            
            save_object(file_path=self.model_trainer_config.train_model_file_path,
                        obj=best_model)
            
            Y_predict = best_model.predict(X_test)
            r2score = r2_score(Y_test,Y_predict)
            
            return r2score
            
            
            
        except Exception as e:
            raise CustomException(e,sys)