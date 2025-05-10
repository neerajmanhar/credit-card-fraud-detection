# Credit Card Fraud Detection

This project is a machine learning-based credit card fraud detection system using a classification model deployed with FastAPI on an AWS EC2 instance. It detects fraudulent credit card transactions by training a model using historical transaction data.


[**AWS Demo Link**](http://13.201.99.47:8000/docs) (This is the public IP address of the EC2 instance)
## Features

- Trains a model using XGBoost or other classifiers for fraud detection.
- Exposes an API endpoint for making predictions using the trained model.
- FastAPI framework for building the API.
- Deployed on AWS EC2 for scalability and access.

## Project Setup

1. **Clone the Repository**

   Clone this repository to your local machine.

   ```bash
   git clone https://github.com/neerajmanhar/credit-card-fraud-detection.git


2. **Create a Virtual Environment**

   Navigate to the project directory and create a virtual environment:

   ```bash
   cd credit-card-fraud-detection
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment**

   On Linux/macOS:

   ```bash
   source venv/bin/activate
   ```

   On Windows:

   ```bash
   .\venv\Scripts\activate
   ```

4. **Install Dependencies**

   Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application Locally**

   You can test the application locally before deploying:

   ```bash
   uvicorn main:app --reload
   ```

   The API will be accessible at `http://localhost:8000`.

## Deployment on AWS EC2

The project is deployed on AWS EC2. You can access the demo API at the following link:

[**AWS Demo Link**](http://13.201.99.47:8000/docs) (This is the public IP address of the EC2 instance)

Make sure to adjust the `host` and `port` as per your deployment settings if necessary.

## File Structure

* `main.py`: Contains the FastAPI application and API routes.
* `model/`: Directory containing the ipynb file.
* `requirements.txt`: List of dependencies required to run the project.
* `README.md`: Documentation for the project.

## Model

The model used for fraud detection is trained on credit card transaction dataset from [**kaggle**](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) and saved as a `.pkl` file for inference.


