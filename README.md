# Waste Classification using Deep Learning

A deep learning model for accurate waste material classification using ResNet50V2. The model classifies 12 different waste categories with 91.36% validation accuracy, demonstrating robust performance in waste identification.

## Model Download

### Pre-trained Model File
- `best_waste_model.h5` (276.03 MB)

### Download Links
- [Google Drive Download Link](https://drive.google.com/drive/folders/1u4ffqv6uoXLssBYE_eEPDKCWjNNJabDG?usp=sharing)

### Download Instructions
1. Click on the Google Drive link above
2. Select "Download" from the Google Drive menu
3. Place the downloaded `best_waste_model.h5` in the project's root directory

## Model Overview

### Architecture
- Base Model: ResNet50V2 with transfer learning
- Input Size: 96x96 pixels
- Training Dataset: 15,515 images
  - Training: 12,412 images
  - Testing: 3,103 images
- Best Validation Accuracy: 91.36%

### Waste Categories
1. Battery
2. Biological
3. Brown Glass
4. Cardboard
5. Clothes
6. Green Glass
7. Metal
8. Paper
9. Plastic
10. Shoes
11. Trash
12. White Glass

## Usage

### Making Predictions
```python
from predict import predict_waste

# Predict waste category for an image
image_path = "path_to_your_image.jpg"
result = predict_waste(image_path)
print(f"Predicted waste type: {result}")
```

### Training the Model
```python
python main.py
```

## Model Architecture Details
```
Model: "waste_classification"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
input                       [(None, 96, 96, 3)]       0         
resnet50v2                  (None, 3, 3, 2048)       23564800  
global_average_pooling2d    (None, 2048)             0         
dense                       (None, 256)               524544    
dropout                     (None, 256)               0         
dense_1                     (None, 12)                3084      
=================================================================
Total params: 24,092,428
Trainable params: 527,628
Non-trainable params: 23,564,800
```

## Technical Specifications
- Framework: TensorFlow 2.x
- Python Version: 3.8+
- Key Libraries: OpenCV, NumPy, Scikit-learn
- Training Features:
  - Data Augmentation
  - Learning Rate Scheduling
  - Early Stopping
  - Dropout Regularization

## Project Structure
```
waste-classification/
├── main.py           # Training implementation
├── predict.py        # Prediction script
├── best_waste_model.h5  # Pre-trained model
├── requirements.txt  # Dependencies
└── README.md        # Documentation
```

## Requirements
```
tensorflow>=2.0.0
opencv-python
numpy
scikit-learn
matplotlib
```

## Performance Metrics
- Training Accuracy: 92.84%
- Validation Accuracy: 91.36%
- Training Time: ~8 hours (CPU)

## Troubleshooting Model Download
- Ensure stable internet connection
- Check Google Drive link is accessible
- Verify downloaded file size (276.03 MB)
- If download fails, try alternative methods or contact project maintainer

## Future Improvements
- Real-time classification support
- Web interface implementation
- Mobile application development
- Model optimization for faster inference

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Author
Parasaram R N V Rajyalakshmi

## Acknowledgments
- Dataset contributors
- Open-source community