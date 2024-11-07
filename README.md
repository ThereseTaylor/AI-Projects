
# 🧠 Machine Learning Projects

Welcome to this collection of machine learning and recommender system projects! This repository includes various projects that uses machine learning and deep learning models for a variety of applications, ranging from image classification to music recommendation and fraud detection.

---

## 📂 Projects Overview

1. **Fruit Classifier (CNN)**:
   - Uses a Convolutional Neural Network (CNN) to classify images of fruits as fresh or rotten.
   - **Key Features**: Deep learning with convolutional layers, data augmentation for generalization.
   - **File**: `CNN.ipynb`
   - **Dataset**: https://zenodo.org/records/7224690

2. **Music Recommender System**:
   - Implements various recommendation algorithms to suggest music tracks to users based on collaborative and content-based filtering.
   - **Key Features**: Collaborative filtering, content-based filtering, recommendation metrics.
   - **File**: `Recommenders.py`

3. **Forest Fire Detection (SVC & Random Forest Models)**:
   - Contains implementations of Support Vector Classifier (SVC) and Random Forest models to detect forest fires based on environmental data.
   - **Key Features**: Multi-model comparison, binary classification.
   - **Files**: `SCV & KNN Models.py`, `RandomForest.py`

4. **Bitcoin Money Laundering Detection (Random Forest Model)**:
   - Uses a Random Forest model to classify Bitcoin transactions as licit or illicit based on risk scores and transaction data.
   - **Key Features**: Risk score calculation, binary classification, fraud detection.
   - **File**: `RF_Model.ipynb`

---

## 🛠 Getting Started

To set up and run these projects, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:

   Install the required libraries using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Projects**:

   Each project can be run independently. Here are the general commands:

   - **For Jupyter Notebooks** (e.g., `CNN.ipynb`, `RF_Model.ipynb`):
     ```bash
     jupyter notebook <notebook_name>.ipynb
     ```

   - **For Python Scripts** (e.g., `Recommenders.py`, `RandomForest.py`, `SCV & KNN Models.py`):
     ```bash
     python <script_name>.py
     ```

---

## 📊 Project Details

### 1. Fruit Classifier (CNN)

This project applies a **Convolutional Neural Network (CNN)** to classify fruit images as either fresh or rotten. The model leverages deep learning techniques, including convolutional and dense layers, to extract features and make predictions on image data.

- **Data Augmentation**: Applies transformations to images to improve model generalization.
- **Evaluation Metrics**: Uses accuracy to assess the model's performance.

### 2. Music Recommender System

This project implements a recommender system to suggest music tracks to users using collaborative and content-based filtering techniques. The system provides personalized music recommendations based on user preferences.

- **Collaborative Filtering**: Recommends items based on user behavior similarity.
- **Content-Based Filtering**: Recommends items based on item attributes and user profiles.
- **Evaluation Metrics**: Includes precision, recall, and ranking metrics.

### 3. Forest Fire Detection (SVC & Random Forest Models)

This project includes Support Vector Classifier (SVC) and Random Forest models to detect forest fires based on environmental data. These machine learning models classify data into fire and non-fire classes to assist in fire detection and prevention.

- **Data Processing**: Prepares environmental data for model input.
- **Model Comparison**: Evaluates SVC and Random Forest models on classification performance.

### 4. Bitcoin Money Laundering Detection (Random Forest Model)

This project applies a Random Forest model to classify Bitcoin transactions as either licit or illicit. The model uses transaction features to calculate a risk score, helping detect potentially illicit activities.

- **Risk Score Calculation**: Generates risk scores to assess transaction legality.
- **Evaluation Metrics**: Uses accuracy and confusion matrix for model evaluation.
