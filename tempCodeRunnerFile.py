from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler (ensure the paths are correct)
model = joblib.load('best_rf_model.pkl')  # Update the path if necessary
scaler = joblib.load('scaler.pkl')  # Load the scaler (if it's saved)

# Define the HTML form inside the Python code for simplicity
form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Form</title>
</head>
<body>
    <h2>Enter Feature Values for Prediction</h2>
    <form method="POST" action="/predict">
        <label for="feature1">Feature 1:</label>
        <input type="number" step="any" name="feature1" required><br><br>

        <label for="feature2">Feature 2:</label>
        <input type="number" step="any" name="feature2" required><br><br>

        <label for="feature3">Feature 3:</label>
        <input type="number" step="any" name="feature3" required><br><br>

        <label for="feature4">Feature 4:</label>
        <input type="number" step="any" name="feature4" required><br><br>

        <input type="submit" value="Predict">
    </form>

    {% if prediction_text %}
        <h3>{{ prediction_text }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(form_html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])

        # Print the received input values for debugging
        print(f"Received input: {feature1}, {feature2}, {feature3}, {feature4}")

        # Combine the features into an array for the model
        features = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)

        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features)

        # Print the scaled features to check the scaling process
        print(f"Scaled features: {features_scaled}")

        # Make the prediction
        prediction = model.predict(features_scaled)[0]

        # Translate prediction to readable result
        result = "High" if prediction == 1 else "Low"

        return render_template_string(form_html, prediction_text=f"Predicted Food Waste Category: {result}")

    except Exception as e:
        # Print the detailed error message for debugging
        print(f"Error: {str(e)}")
        return render_template_string(form_html, prediction_text="Error occurred. Please check your inputs.")

if __name__ == '__main__':
    app.run(debug=True)
