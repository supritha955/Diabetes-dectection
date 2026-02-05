# Diabetes Prediction Web App

This project is a full-stack Django app that trains ML models to predict diabetes and exposes a REST API for real-time predictions.

Quick start:

1. Create and activate a Python virtual environment (Windows):
   python -m venv venv
   venv\Scripts\activate
2. Install dependencies:
   pip install -r requirements.txt
3. Put the dataset CSV `Diabetes detection.csv` in the project root or `dataset/` folder.
4. Train models (this will generate model artifacts and images):
   python diabetes_project\prediction\ml_model.py
5. Run migrations and start server:
   python manage.py migrate
   python manage.py runserver
6. Visit http://127.0.0.1:8000/ and use the form to get predictions.

Notes:
- The API endpoint is `POST /api/predict/` which accepts JSON with the numeric features.
- For simplicity the API view is CSRF-exempt; for production, secure it properly.

