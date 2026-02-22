from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load model and classes based on your dataset
try:
    model = tf.keras.models.load_model('waste_classifier.h5')
    print("✅ Model loaded successfully!")
except:
    print("❌ No trained model found. Please run train_model.py first")
    model = None

# Your dataset classes
classes = ['biodegradable', 'recyclable', 'trash']

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        image = Image.open(filepath).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = classes[predicted_class_idx]
        confidence = float(np.max(predictions)) * 100
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'image_path': f'/{filepath}',
            'all_predictions': {
                classes[i]: round(float(predictions[0][i]) * 100, 2) 
                for i in range(len(classes))
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)