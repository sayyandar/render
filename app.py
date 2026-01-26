import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import traceback
import os

app = Flask(__name__)

# Try multiple loading strategies for maximum compatibility
def load_model_with_fallback():
    """Try loading model in different formats"""
    model = None
    scaler = None
    
    # Try loading scaler first
    try:
        scaler = joblib.load('saved_model/scaler.pkl')
        print("✅ Scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        scaler = None
    
    # Try different model formats in order of preference
    model_formats = [
        ('saved_model/neural_network_model.keras', 'Keras .keras format'),
        ('saved_model/neural_network_model.h5', 'Keras .h5 format'),
        ('saved_model/lightweight_model.keras', 'Lightweight model'),
    ]
    
    for model_path, format_name in model_formats:
        try:
            if os.path.exists(model_path):
                from tensorflow import keras
                model = keras.models.load_model(model_path)
                print(f"✅ Model loaded from {format_name}")
                break
            else:
                print(f"⚠️  Model file not found: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load {format_name}: {e}")
    
    # Fallback: Load from JSON + weights
    if model is None:
        try:
            from tensorflow import keras
            with open('saved_model/model_architecture.json', 'r') as f:
                model_arch = f.read()
            model = keras.models.model_from_json(model_arch)
            model.load_weights('saved_model/model_weights.h5')
            print("✅ Model loaded from JSON + weights")
        except Exception as e:
            print(f"❌ Failed to load from JSON + weights: {e}")
    
    return model, scaler

# Load model and scaler at startup
print("=" * 60)
print("🚀 Starting Phishing URL Detector")
print("=" * 60)

model, scaler = load_model_with_fallback()

if model is None:
    print("❌ CRITICAL: Could not load any model format!")
    print("Please run fix_model_compatibility.py first")
if scaler is None:
    print("⚠️  WARNING: Could not load scaler")
    print("Predictions may not be accurate")

print("=" * 60)

# Feature extraction function (keep as is)
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
        prediction_prob = model.predict(features_scaled, verbose=0)[0][0]
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
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Extract and predict
        features = extract_features_from_url(url)
        features_scaled = scaler.transform(features)
        prediction_prob = model.predict(features_scaled, verbose=0)[0][0]
        
        return jsonify({
            'url': url,
            'phishing_score': float(prediction_prob),
            'is_phishing': prediction_prob > 0.5,
            'confidence': f"{prediction_prob * 100:.2f}%",
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    if model and scaler:
        return jsonify({
            'status': 'healthy', 
            'model_loaded': True,
            'scaler_loaded': True
        })
    return jsonify({
        'status': 'unhealthy', 
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }), 500

@app.route('/model-info')
def model_info():
    """Get model information endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        model_type = type(model).__name__
        
        if hasattr(model, 'input_shape'):
            info = {
                'model_type': model_type,
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'layers': len(model.layers),
                'status': 'loaded'
            }
        else:
            info = {
                'model_type': model_type,
                'status': 'loaded'
            }
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/robots.txt')
def robots():
    return send_from_directory('static', 'robots.txt')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting server on port {port}")
    print(f"Debug mode: {debug}")
    
    # For production, use 0.0.0.0 to accept connections from all IPs
    app.run(host='0.0.0.0', port=port, debug=debug)