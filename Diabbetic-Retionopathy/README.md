
# Diabetic Retinopathy Detection Using CNN

This project focuses on detecting **Diabetic Retinopathy (DR)** from retinal fundus images using a **Convolutional Neural Network (CNN)**.  
The goal is to assist early diagnosis by classifying retinal images into different severity levels.

##  Project Structure
├── trained_model/
│ └── trained_model.h5
├── notebooks/
│ └── model_training.ipynb
├── report.pdf
├── requirements.txt
└── README.md

---

##  Objective

To train a deep learning model that classifies retinal images into **5 Diabetic Retinopathy classes** and provides **model explainability using Grad-CAM**.

##  Classes

- Class 0: No DR  
- Class 1: Mild DR  
- Class 2: Moderate DR  
- Class 3: Severe DR  
- Class 4: Proliferative DR  

## Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib  

## Dataset & Preprocessing

- Images loaded using `image_dataset_from_directory`
- Image size: **64 × 64**
- Data augmentation:
  - Rescaling
  - Horizontal flip
  - Rotation
  - Zoom
- Dataset split into **train** and **validation**
##  Model Architecture

- 3 Convolutional layers with Batch Normalization
- MaxPooling & Dropout for regularization
- Global Average Pooling
- Dense layers
- Softmax output for 5 classes
##  Training Details

- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Epochs: 5  
- Early Stopping applied  

## Evaluation

Model evaluated using:
- Accuracy
- Precision
- Recall
- F1-score  

Classification report is generated on validation data.

##  Explainability (Grad-CAM)

Grad-CAM is implemented to visualize important regions of retinal images influencing model predictions.

##  Trained Model

Saved model file:
trained_model/trained_model.h5

##  Report

Detailed explanation of methodology, training, results, and Grad-CAM visualizations is available in **report.pdf**.




