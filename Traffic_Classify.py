import os
import numpy as np
import cv2
import pickle
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn import svm
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

# Read and load model data
def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir('Traffic-Data/trainingset'):
        for img_file in os.listdir(os.path.join('Traffic-Data/trainingset', label)):
            img = cv2.imread(os.path.join('Traffic-Data/trainingset', label, img_file))
            X.append(img)
            Y.append(label2id[label])
    return X, Y

label2id = {'pedestrian': 0, 'moto': 1, 'truck': 2, 'car': 3, 'bus': 4}
X, Y = read_data(label2id)

# Extract SIFT features
def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.SIFT_create()
    for img in X:
        kp, des = sift.detectAndCompute(img, None)
        image_descriptors.append(des)
    return image_descriptors

image_descriptors = extract_sift_features(X)

# Build BoW dictionary using KMeans
def kmeans_bow(all_descriptors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    return kmeans.cluster_centers_

all_descriptors = [des for des_list in image_descriptors if des_list is not None for des in des_list]
num_clusters = 100
if not os.path.isfile('Traffic-Data/bow_dictionary150.pkl'):
    BoW = kmeans_bow(all_descriptors, num_clusters)
    pickle.dump(BoW, open('Traffic-Data/bow_dictionary150.pkl', 'wb'))
else:
    BoW = pickle.load(open('Traffic-Data/bow_dictionary150.pkl', 'rb'))

# Convert images to feature vectors
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    for des in image_descriptors:
        features = np.zeros(num_clusters)
        if des is not None:
            distances = cdist(des, BoW)
            closest_clusters = np.argmin(distances, axis=1)
            for cluster_idx in closest_clusters:
                features[cluster_idx] += 1
        X_features.append(features)
    return X_features

X_features = create_features_bow(image_descriptors, BoW, num_clusters)
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = svm.SVC(C=10)
svm_model.fit(X_train, Y_train)

# Tkinter GUI
class TrafficClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Image Classifier")
        
        # Select Image Button
        self.btn_select = tk.Button(root, text="Select Image", command=self.select_image)
        self.btn_select.pack(pady=10)

        # Predict Button
        self.btn_predict = tk.Button(root, text="Predict", command=self.predict_image)
        self.btn_predict.pack(pady=10)

        # Label for displaying predictions
        self.label_prediction = tk.Label(root, text="Prediction: ")
        self.label_prediction.pack(pady=10)

        # Canvas to display selected image
        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.pack(pady=10)

        self.file_path = None

    def select_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            img = Image.open(self.file_path)
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(150, 150, image=img_tk)
            self.canvas.image = img_tk

    def predict_image(self):
        if not self.file_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        img = cv2.imread(self.file_path)
        img_descriptors = extract_sift_features([img])
        img_features = create_features_bow(img_descriptors, BoW, num_clusters)

        prediction = svm_model.predict(img_features)
        prediction_label = [key for key, value in label2id.items() if value == prediction[0]][0]
        
        self.label_prediction.config(text=f"Prediction: {prediction_label}")
        
# Run the application
root = tk.Tk()
app = TrafficClassifierApp(root)
root.mainloop()
