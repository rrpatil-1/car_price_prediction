FROM python:3.12
WORKDIR /app
COPY . /app
EXPOSE 8000
RUN pip install -r requirements.txt
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]