import os
import pickle
import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.auth import exceptions

# Flask setup
app = Flask(__name__)

# Google Drive API setup (if you want to download files from Google Drive)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']  # Set the scope to read files
CLIENT_SECRETS_FILE = 'client_secrets.json'  # Path to your client_secrets.json file

# Global variables
credentials = None
MODEL_FILE_ID = 'your_model_file_id'  # Replace with your model file ID on Google Drive
model_path = 'best_rf_model.pkl'  # Local path to save the model (update this)

# Check if credentials are already saved, if not, authenticate
def get_credentials():
    global credentials

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)

    # If credentials are invalid or not present, perform OAuth 2.0 authentication
    if not credentials or credentials.expired or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_local_server(port=0)

        # Save credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)

    return credentials

# Google Drive authentication and file download
def download_file_from_drive(file_id, local_path):
    try:
        credentials = get_credentials()
        service = build('drive', 'v3', credentials=credentials)
        
        # Request the file from Google Drive
        request = service.files().get_media(fileId=file_id)
        fh = open(local_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        
        fh.close()
        print(f"File downloaded successfully to {local_path}")

    except exceptions.GoogleAuthError as auth_error:
        print(f"Authentication error: {auth_error}")
    except Exception as error:
        print(f"Error downloading file: {error}")

# Load the model (locally or from Google Drive)
def load_model():
    if not os.path.exists(model_path):
        download_file_from_drive(MODEL_FILE_ID, model_path)
    return joblib.load(model_path)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        features = [float(request.form[key]) for key in request.form.keys()]
        features = np.array(features).reshape(1, -1)
        
        # Load the model and make the prediction
        model = load_model()
        scaler = StandardScaler()  # Ensure the scaler matches the one used in training
        features = scaler.fit_transform(features)
        prediction = model.predict(features)[0]
        
        # Translate the prediction result
        result = "High" if prediction == 1 else "Low"
        
        return render_template('index.html', prediction_text=f"Predicted Food Waste Category: {result}")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text="Error occurred. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
