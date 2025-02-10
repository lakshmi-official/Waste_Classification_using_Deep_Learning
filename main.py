import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class WasteImagePreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.categories = [
            'battery', 'biological', 'brown-glass', 'cardboard',
            'clothes', 'green-glass', 'metal', 'paper',
            'plastic', 'shoes', 'trash', 'white-glass'
        ]

    def load_and_preprocess_images(self, img_size=(96, 96)):
        images = []
        labels = []
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )

        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            category_index = self.categories.index(category)

            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)

                try:
                    img = cv2.imread(img_path)
                    img_resized = cv2.resize(img, img_size)
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_preprocessed = preprocess_input(img_rgb)

                    # Apply data augmentation
                    img_aug = datagen.random_transform(img_preprocessed)

                    images.append(img_aug)
                    labels.append(category_index)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        return np.array(images), np.array(labels)

    def create_tensorflow_dataset(self, X_train, y_train, X_test, y_test, batch_size=32):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        return train_dataset, test_dataset

def create_waste_classification_model(input_shape=(96, 96, 3), num_classes=12):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def lr_schedule(epoch):
    return 1e-4 * (0.9 ** epoch)

def train_model(train_dataset, test_dataset, epochs=50):
    model = create_waste_classification_model()
    model.summary()

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_waste_model.h5', save_best_only=True, monitor='val_accuracy'),
        lr_scheduler
    ]

    history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=callbacks)

    # Fine-tuning
    base_model = model.layers[1]
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    fine_tune_history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=callbacks)

    # Combine histories
    total_history = {}
    for k in history.history.keys():
        total_history[k] = history.history[k] + fine_tune_history.history[k]

    return model, total_history

def evaluate_model(model, test_dataset):
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    return test_loss, test_accuracy

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

def main():
    dataset_path = 'Datasets'
    preprocessor = WasteImagePreprocessor(dataset_path)
    images, labels = preprocessor.load_and_preprocess_images(img_size=(96, 96))

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
    train_dataset, test_dataset = preprocessor.create_tensorflow_dataset(X_train, y_train, X_test, y_test)

    print("Total images:", len(images))
    print("Training images:", len(X_train))
    print("Image for testing images:", len(X_test))
    print("Categories:", preprocessor.categories)

    model, history = train_model(train_dataset, test_dataset)
    test_loss, test_accuracy = evaluate_model(model, test_dataset)
    plot_training_history(history)

    model.save('final_waste_classification_model.h5')
    print("Model saved successfully!")

if __name__ == '__main__':
    main()