from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model (without Present_Price)
model = pickle.load(open('rfr_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Numerical inputs
    year = int(request.form['year'])
    kms_driven = int(request.form['kms_driven'])
    owner = int(request.form['owner'])

    # Categorical inputs
    fuel = request.form['fuel']
    seller = request.form['seller']
    transmission = request.form['transmission']

    # Feature engineering
    car_age = 2025 - year

    # Encoding (MUST match training)
    fuel_diesel = 1 if fuel == 'Diesel' else 0
    seller_individual = 1 if seller == 'Individual' else 0
    transmission_manual = 1 if transmission == 'Manual' else 0

    # FINAL FEATURE ORDER (VERY IMPORTANT)
    final_features = np.array([[
        year,
        kms_driven,
        owner,
        car_age,
        fuel_diesel,
        seller_individual,
        transmission_manual
    ]])

    # Prediction
    prediction = model.predict(final_features)[0]
    prediction = max(0, prediction)  # avoid negative output

    return render_template(
        'index.html',
        prediction_text=f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs"
    )

@app.errorhandler(500)
def internal_error(error):
    return "Internal Server Error", 500

if __name__ == "__main__":
    app.run(debug=True)
