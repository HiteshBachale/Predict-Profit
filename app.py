# Import necessary libraries
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # This is crucial for handling requests from the HTML file
import joblib  # or import pickle if you used pickle to save your model
import numpy as np

# Create the Flask application instance
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the path to your trained model file
# IMPORTANT: Replace 'your_model.pkl' with the actual name of your model file.
MODEL_FILE = 'MLR_Model.pkl'

# Load the model once when the application starts
try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found. Please make sure it's in the same directory.")
    model = None  # Set model to None to prevent errors


@app.route('/')
def home():
    """
    Renders the main HTML page.
    For this setup, you can simply have your HTML file
    as a static file or serve it directly.
    """
    return "Please open the index.html file in your browser to use the application."


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the front end.
    """
    # Check if the model was loaded successfully
    if model is None:
        return jsonify({'error': 'Model not found.'}), 500

    # Get the JSON data from the POST request
    data = request.get_json(force=True)

    # Extract the input values from the received data
    rd_spend = data['rd_spend']
    admin_spend = data['admin_spend']
    marketing_spend = data['marketing_spend']
    state = data['state']

    # Create a NumPy array with the input data.
    # The order of the features must match the order the model was trained on.
    # Assuming R&D, Administration, Marketing, and State are the features in that order.
    # The reshape(-1, 4) is important for a single prediction.
    features = np.array([[rd_spend, admin_spend, marketing_spend, state]])

    # Make the prediction
    prediction = model.predict(features)

    # The prediction is a numpy array, so we extract the single value
    predicted_profit = prediction[0]

    # Return the prediction as a JSON response
    return jsonify({
        'predicted_profit': predicted_profit
    })


if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
