from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import CustomLogger

logger = CustomLogger()

app = FastAPI()

# Create a Pydantic model for request validation
class PredictionInput(BaseModel):
    JobTitle: str
    country: str
    gender: str
    age: int
    BasePay: float
    OvertimePay: float
    OtherPay: float
    TotalPay: float
    TotalPayBenefits: float
    CreditCardDebt: float
    NetWorth: float

    class Config:
        json_schema_extra = {
            "example": {
                "JobTitle": "TRANSIT SUPERVISOR",
                "country": "Nepal",
                "gender": "M",
                "age": 30,
                "BasePay": 75000.0,
                "OvertimePay": 5000.0,
                "OtherPay": 2000.0,
                "TotalPay": 82000.0,
                "TotalPayBenefits": 85000.0,
                "CreditCardDebt": 5000.0,
                "NetWorth": 150000.0
            }
        }

@app.post("/predictscore")
async def predictscore(input_data: PredictionInput):
    try:
        logger.log(f"request received: {input_data.dict()}")
        
        data = CustomData(
            JobTitle=input_data.JobTitle,
            country=input_data.country,
            gender=0 if input_data.gender=="M" else 1,
            age=input_data.age,
            BasePay=input_data.BasePay,
            OvertimePay=input_data.OvertimePay,
            OtherPay=input_data.OtherPay,
            TotalPay=input_data.TotalPay,
            TotalPayBenefits=input_data.TotalPayBenefits,
            CreditCardDebt=input_data.CreditCardDebt,
            NetWorth=input_data.NetWorth        )
        
        pred_df = data.get_data_as_data_frame()
        logger.log(pred_df)
        
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        
        return {"ExpectedCarPurchaseAmount": float(pred[0])}
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
