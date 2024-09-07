from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Log when the app starts
print("App starting...")

# Load the trained model from the .pkl file
try:
    model_path = os.path.join(os.getcwd(), 'model.pkl')  # Make sure the model is in the correct path
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log request reception
        print("Prediction request received.")

        # Get the data from the POST request
        data = request.get_json(force=True)
        print("Received data:", data)

        # Convert data into a format suitable for the model
        input_data = np.array([data['input']])
        print("Input data for prediction:", input_data)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        print("Prediction result:", prediction)

        # Return the prediction as a JSON response
        return jsonify({'prediction': str(prediction[0])})
    except Exception as e:
        # Log the error details
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred while processing the prediction', 'message': str(e)}), 500
