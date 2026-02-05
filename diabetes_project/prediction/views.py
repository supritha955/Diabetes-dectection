import json
import os
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / 'prediction' / 'models' / 'best_model.joblib'
SCALER_PATH = BASE_DIR / 'prediction' / 'models' / 'scaler.joblib'


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def api_predict(request):
    if request.method != 'POST':
        return HttpResponseBadRequest('Only POST allowed')
    try:
        data = json.loads(request.body)
    except Exception:
        return HttpResponseBadRequest('Invalid JSON')

    required_features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]

    try:
        values = [float(data[f]) for f in required_features]
    except Exception:
        return HttpResponseBadRequest('Missing or invalid features')

    # Basic validation
    if any(v < 0 for v in values):
        return HttpResponseBadRequest('Feature values must be non-negative')

    # Load objects
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return HttpResponseBadRequest('Model not trained yet. Run training script.')

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = None
    try:
        prob = model.predict_proba(X_scaled)[0]
        confidence = float(round(max(prob) * 100, 2))
    except Exception:
        confidence = None

    label = 'Diabetic' if int(pred) == 1 else 'Non-Diabetic'

    return JsonResponse({'prediction': label, 'confidence': confidence})


def visuals(request):
    # Returns URLs for visualization images
    base_static = '/static/images/'
    return JsonResponse({
        'confusion_matrix': base_static + 'confusion_matrix.png',
        'accuracy_comparison': base_static + 'accuracy_comparison.png',
    })
