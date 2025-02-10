from tensorflow.keras.models import load_model
import cv2
import numpy as np

def predict_waste(image_path):
    model = load_model('best_waste_model.h5')
    categories = ['battery', 'biological', 'brown-glass', 'cardboard',
                 'clothes', 'green-glass', 'metal', 'paper',
                 'plastic', 'shoes', 'trash', 'white-glass']

    img = cv2.imread(image_path)
    img = cv2.resize(img, (96, 96))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return categories[np.argmax(prediction)]


image_path = "Image for testing/img.png"
result = predict_waste(image_path)
print(f"This is: {result}")