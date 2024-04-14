import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
            images.append(img.flatten())  # Flatten the image
            filenames.append(filename)
    return images, filenames


def load_labeled_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
            images.append(img.flatten())  # Flatten the image
            labels.append(label)
    return images, labels


# Load training dataset
train_folder_cats = 'train/cats'
train_folder_dogs = 'train/dogs'
train_cats_data, train_cats_labels = load_labeled_images_from_folder(train_folder_cats, 0)
train_dogs_data, train_dogs_labels = load_labeled_images_from_folder(train_folder_dogs, 1)
print('Loading training dataset Done\n')

# Combine training data
sample_size=5000
X_train = np.array(train_cats_data[:sample_size] + train_dogs_data[:sample_size])
y_train = np.array(train_cats_labels [:sample_size]+ train_dogs_labels[:sample_size])
X_train, y_train = shuffle(X_train, y_train, random_state=42)
print('Combine training data and shuffle Done\n')

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Load testing dataset
test_folder = 'test'  # Folder containing unlabeled test data
X_test, test_filenames = load_images_from_folder(test_folder[:sample_size])
print('Load testing dataset Done\n')

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
print('Scaling features Done\n')

# Train the SVM
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
print('Training the SVM Done\n')

# Predict on the test set
y_pred_val = svm_classifier.predict(X_val)
y_pred = svm_classifier.predict(X_test[:100])
print('Predicting the test set Done\n')

# Evaluate the classifier on the validation set
accuracy = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy)
print("\nValidation Classification Report:\n", classification_report(y_val, y_pred_val))

# Output predictions with filenames
for filename, label in zip(test_filenames, y_pred):
    print(f"Filename: {filename}, Predicted Label: {'Cat' if label == 0 else 'Dog'}")
print('Finished')