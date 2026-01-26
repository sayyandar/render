import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import traceback
import os

# Remove static_folder='.' - this is wrong!
app = Flask(__name__)  # Let Flask use default 'static' folder

# Load the saved model and scaler
try:
    # Use absolute path for Render
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'saved_model', 'neural_network_model.h5')
    scaler_path = os.path.join(base_dir, 'saved_model', 'scaler.pkl')
    
    # Try to load TensorFlow model
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        print("✅ TensorFlow model loaded successfully!")
    except ImportError:
        print("⚠️ TensorFlow not available, using fallback")
        model = None
    except Exception as e:
        print(f"⚠️ Could not load TensorFlow model: {e}")
        model = None
    
    scaler = joblib.load(scaler_path)
    print("✅ Scaler loaded successfully!")
    
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
        return jsonify({
            'error': 'Model not loaded properly',
            'url': request.form.get('url', ''),
            'is_phishing': True,  # Default to safe
            'phishing_probability': 0.5,
            'confidence': "50.0%",
            'safe_probability': "50.0%",
            'model_status': 'fallback'
        })
    
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
        prediction_prob = model.predict(features_scaled, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        # Prepare result
        result = {
            'url': url,
            'is_phishing': bool(prediction),
            'phishing_probability': float(prediction_prob),
            'confidence': f"{prediction_prob * 100:.2f}%",
            'safe_probability': f"{(1 - prediction_prob) * 100:.2f}%",
            'model_status': 'neural_network'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    if model and scaler:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

# Optional: Serve favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)