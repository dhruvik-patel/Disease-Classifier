# Chest X-Ray Disease Prediction Web App

This is a Flask web application that uses a trained PyTorch model to predict diseases from chest X-ray images.

## Features

- Upload X-ray images via drag-and-drop or file selector
- Preview uploaded images before prediction
- Disease prediction with probability percentages
- Visual results with plots showing the original image and probability distribution

## Setup Instructions

1. Ensure you have Python installed (version 3.7 or higher recommended)

2. Activate the virtual environment:
   ```
   .\venv\Scripts\activate
   ```

3. Install the required packages (if not already done):
   ```
   pip install flask torch torchvision pillow numpy matplotlib
   ```

4. Place your trained model in the `model` directory:
   ```
   model/best_model.pth
   ```
   Note: You need to have a trained model. The model should be a PyTorch checkpoint file.

5. Run the Flask application:
   ```
   python app.py
   ```

6. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Using the Web App

1. Drag and drop a chest X-ray image onto the upload area or click the "Select File" button to choose an image file.
2. After uploading, a preview of the image will be displayed.
3. Click the "Predict Diseases" button to analyze the image.
4. Results will show on the right side, including:
   - A visual plot of the X-ray and disease probabilities
   - A table listing all diseases and their predicted probabilities

## Model Information

The model uses a DenseNet121 architecture pre-trained on ImageNet and fine-tuned for multi-label classification of 14 different lung diseases commonly found in chest X-rays:

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia

## Troubleshooting

If you encounter any issues:

- Make sure your model file is correctly placed in the `model` directory
- Check that the model architecture matches what's expected in `model.py`
- Ensure your image is in a supported format (JPG, JPEG, PNG)
- Check the Flask server logs for any error messages

## License

This project is provided for educational purposes only. Use responsibly. 