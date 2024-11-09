from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from waitress import serve  # Import waitress to serve the Flask app

app = Flask(__name__)

# Load the saved model and initialize the scaler
model = joblib.load('best_rf_model.pkl')  # Provide the correct path to your model file
scaler = StandardScaler()  # Ensure the scaler matches the one used in training

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and process input data from the form
        features = [float(request.form[key]) for key in request.form.keys()]
        features = np.array(features).reshape(1, -1)
        
        # Scale the features
        features = scaler.fit_transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Translate prediction result to a readable output
        result = "High" if prediction == 1 else "Low"
        
        return render_template('index.html', prediction_text=f"Predicted Food Waste Category: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text="Error occurred. Please try again.")

# Use Waitress to serve the app (instead of Flask's built-in server)
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)  # You can change the host/port as needed
