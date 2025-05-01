from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and train the model only once when server starts
model = None

def train_model():
    df = pd.read_csv('app1/Dataset/heart_disease_data.csv')  # Check correct path
    X = df[['age', 'cp', 'thalach']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trained_model = LogisticRegression()
    trained_model.fit(X_train, y_train)
    return trained_model

# Prediction function
def predict_heart_disease(model, age, cp, thalach):
    user_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])
    prediction = model.predict(user_data)
    if prediction[0] == 1:
        return "⚠️ High Risk: The person has heart disease"
    else:
        return "✅ Low Risk: The person does not have heart disease"

# View function
def home(request):
    global model
    if model is None:
        model = train_model()  # Load model once

    submitted_data = None
    prediction_result = None

    if request.method == 'POST':
        try:
            age = int(request.POST.get('age'))
            chest_pressure = int(request.POST.get('chest_pressure'))
            lh = int(request.POST.get('lh'))

            submitted_data = {
                'age': age,
                'chest_pressure': chest_pressure,
                'lh': lh,
            }

            prediction_result = predict_heart_disease(model, age, chest_pressure, lh)
        
        except (ValueError, TypeError):
            prediction_result = "Invalid input. Please enter correct numeric values."

    return render(request, 'patient_form.html', {
        'submitted_data': submitted_data,
        'prediction_result': prediction_result,
    })
