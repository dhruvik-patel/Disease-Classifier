import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import model
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Set the path to your model
MODEL_PATH = 'model/best_model.pth'

# Initialize the model
try:
    chest_xray_model = model.load_model(MODEL_PATH)
    print("Model loaded successfully")
except FileNotFoundError:
    print(f"Warning: Model file not found at {MODEL_PATH}. Please place your model in this location.")
    print("Running with no model for now. Upload will still work but predictions will fail.")
    chest_xray_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read and process the image
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # Check if model is loaded
            if chest_xray_model is None:
                return jsonify({'error': 'Model not loaded. Please ensure the model file is in the correct location.'}), 500
            
            # Make prediction
            predictions = model.predict_image(chest_xray_model, img)
            
            # Generate plot
            plot_base64 = model.generate_prediction_plot(img, predictions)
            
            # Format results for display
            results = []
            for disease, prob in predictions:
                results.append({
                    'disease': disease,
                    'probability': f"{prob*100:.2f}%"
                })
            
            return jsonify({
                'success': True,
                'predictions': results,
                'plot': plot_base64
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    # Create the model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    app.run(debug=True) 