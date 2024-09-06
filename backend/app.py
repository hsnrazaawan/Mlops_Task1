from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
import joblib
import numpy as np

app = Flask(__name__)

# Explicitly allow only your frontend origin
CORS(app, origins=["https://mlops-frontend.vercel.app"])

# Load the trained model from the .pkl file
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        print("Received data:", data)

        # Convert data into a format suitable for the model (ensure it's a 2D array)
        input_data = np.array([data['input']])
        print("Input data for prediction:", input_data)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        print("Prediction result:", prediction)

        # Return the prediction as a JSON response
        return jsonify({'prediction': str(prediction[0])})  # Assuming it's a single value prediction
    except Exception as e:
        # Log and return an error message
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred while processing the prediction', 'message': str(e)}), 500
