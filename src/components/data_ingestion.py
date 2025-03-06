import os,sys
from pathlib import Path

# Get the absolute path of the project root directory
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
from src.exception import CustomException
from src.logger import CustomLogger
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer

logger =CustomLogger()

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifact','train.csv')
    test_data_path:str = os.path.join('artifact','test.csv')
    raw_data_path:str = os.path.join('artifact','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('entered data ingestion method/ component')
        try:
            df = pd.read_csv('notebook/dataset/car_price_purchasing_v1.csv')
            
            logging.info('read data as dataframe')
            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path),exist_ok=True)
            df.to_csv(self.ingestionconfig.raw_data_path,index=False)
            
            logging.info('train test initiate')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestionconfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path,index=False,header=True)
            
            logging.info('ingestion of data completed')
            
            return (
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj =  DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr=data_transformation.initiate_data_transofrmation(train_data_path,test_data_path)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    