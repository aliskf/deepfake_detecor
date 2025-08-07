
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os


model = tf.keras.models.load_model('deepfake_detector_model.h5')


IMG_WIDTH, IMG_HEIGHT = 128, 128


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


real_images_dir = 'image_classification_data/real'
fake_images_dir = 'image_classification_data/fake'

real_image_files = [os.path.join(real_images_dir, f) for f in os.listdir(real_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
fake_image_files = [os.path.join(fake_images_dir, f) for f in os.listdir(fake_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

all_image_paths = real_image_files + fake_image_files
all_labels = [1] * len(real_image_files) + [0] * len(fake_image_files) # 1 for real, 0 for fake

predictions = []
true_labels = []

print("--- Evaluating Model Performance ---")

for i, img_path in enumerate(all_image_paths):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img, verbose=0)[0][0] # Get the single prediction value

    predictions.append(prediction)
    true_labels.append(all_labels[i])

    image_name = os.path.basename(img_path)
    predicted_class = "REAL" if prediction > 0.5 else "FAKE"
    true_class = "REAL" if all_labels[i] == 1 else "FAKE"
    confidence = prediction if predicted_class == "REAL" else (1 - prediction)

    print(f'{image_name}: True: {true_class}, Predicted: {predicted_class} (Confidence: {confidence:.2f})')

# Calculate metrics
binary_predictions = [1 if p > 0.5 else 0 for p in predictions] # 0 for real, 1 for fake

correct_predictions = np.sum(np.array(binary_predictions) == np.array(true_labels))
total_predictions = len(true_labels)
accuracy = correct_predictions / total_predictions

print(f"\nTotal images evaluated: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}")

print("\n--- Evaluation Complete ---")
