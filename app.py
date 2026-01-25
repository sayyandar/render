import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
# pyright: ignore[reportMissingImports]
from tensorflow.keras.models import load_model 
import traceback

app = Flask(__name__)

# Load the saved model and scaler
try:
    model = load_model('saved_model/neural_network_model.h5')
    scaler = joblib.load('saved_model/scaler.pkl')
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(traceback.format_exc())
    model = None
    scaler = None

# Feature names (should match your training data)
FEATURE_NAMES = [
    'url_length', 'n_dots', 'n_hyphens', 'n_underline', 'n_slash', 
    'n_question', 'n_equal', 'n_at', 'n_and', 'n_exclamation', 
    'n_space', 'n_tilde', 'n_comma', 'n_plus', 'n_asterisk', 
    'n_hashtag', 'n_dollar', 'n_percent', 'n_redirection'
]

def extract_features_from_url(url):
    """Extract 19 features from a URL"""
    features = []
    
    # 1. URL Length
    features.append(len(url))
    
    # 2-19. Count specific characters
    characters_to_count = [
        '.', '-', '_', '/', '?', '=', '@', '&', 
        '!', ' ', '~', ',', '+', '*', '#', '$', '%'
    ]
    
    for char in characters_to_count:
        features.append(url.count(char))
    
    # 20. Count redirections (simplified - count occurrences of http/https)
    features.append(url.count('http'))
    
    return np.array(features).reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/phishing-examples')
def phishing_examples():
    return render_template('phishing_examples.html')

@app.route('/safety-tips')
def safety_tips():
    return render_template('safety_tips.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        # Get URL from form
        url = request.form.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'Please provide a URL'}), 400
        
        # Extract features
        features = extract_features_from_url(url)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction_prob = model.predict(features_scaled)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        # Prepare result
        result = {
            'url': url,
            'is_phishing': bool(prediction),
            'phishing_probability': float(prediction_prob),
            'confidence': f"{prediction_prob * 100:.2f}%",
            'safe_probability': f"{(1 - prediction_prob) * 100:.2f}%"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Extract and predict
        features = extract_features_from_url(url)
        features_scaled = scaler.transform(features)
        prediction_prob = model.predict(features_scaled)[0][0]
        
        return jsonify({
            'url': url,
            'phishing_score': float(prediction_prob),
            'is_phishing': prediction_prob > 0.5,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    if model and scaler:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)