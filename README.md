# Car Price Prediction

This project aims to predict car prices using machine learning models.

## Getting Started

### Prerequisites
- python 3.12

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/rrpatil-1/car_price_prediction.git
    
    cd car_price_prediction
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Project Structure
-------------------------------------------------------- 
📂 cd car_price_prediction
- 📂 artifact
  - data.csv
  - model.pkl
  - preprocessor.pkl
  - test.csv
  - train.csv

- 📂 notebook
  - 📂 dataset
    - 📄 car_price_purchasing_v1.csv
    - 📄 car_price_purchasing.csv  
      
          This  contain raw data and clean data

  - 📄 EDA.ipynb
  - 📄 MODEL_TRAINING.ipynb
 - 📂 src
   - 📂 components
     - 📄 \___init___.py
     - 📄 data_ingestion.py
     - 📄 data_transformation.py
     - 📄 model_trainer.py
   - 📂 pipeline
     - 📄 \___init___.py
     - 📄 predict_pipeline.py
   - 📄 exception.py
   - 📄 logger.py
   - 📄 utils.py
    
- 📄 application.py

## Usage

To run the application, execute :
```sh
uvicorn application:app
```
## How to get prediction result 
visit the url
http://localhost:8000/docs


# Prediction Score API

This API predicts the purchase score based on employee compensation details. It accepts inputs related to salary components, job information, demographics, and more.

## API Details

- **Endpoint:** `/predictscore`
- **Method:** `POST`
- **Body:**

```json
{
  "BasePay": 75000,
  "CreditCardDebt": 5000,
  "JobTitle": "TRANSIT SUPERVISOR",
  "NetWorth": 150000,
  "OtherPay": 2000,
  "OvertimePay": 5000,
  "TotalPay": 82000,
  "TotalPayBenefits": 85000,
  "age": 30,
  "country": "Nepal",
  "gender": "M"
}
```
## Field Documentation
| Field              | Value                                                | Datatype |
|--------------------|------------------------------------------------------|----------|
| BasePay            | Numeric value representing the base salary/pay       | number   |
| CreditCardDebt     | Numeric value representing credit card debt          | number   |
| JobTitle           | Job title (e.g., "TRANSIT SUPERVISOR")                 | string   |
| NetWorth           | Numeric value representing net worth                 | number   |
| OtherPay           | Numeric value representing other pay components      | number   |
| OvertimePay        | Numeric value representing overtime pay              | number   |
| TotalPay           | Numeric value representing total pay                 | number   |
| TotalPayBenefits   | Numeric value representing total pay including benefits| number   |
| age                | Age of the individual                                | number   |
| country            | Country name                                         | string   |
| gender             | Gender (e.g., "M" or "F")                              | string   |


### python code for testing api
```python
import requests

url = "http://127.0.0.1:5000/predictscore"

body = {
    "BasePay": 75000,
    "CreditCardDebt": 5000,
    "JobTitle": "TRANSIT SUPERVISOR",
    "NetWorth": 150000,
    "OtherPay": 2000,
    "OvertimePay": 5000,
    "TotalPay": 82000,
    "TotalPayBenefits": 85000,
    "age": 30,
    "country": "Nepal",
    "gender": "M"
}

response = requests.post(url, json=body)
print(response.text)
```