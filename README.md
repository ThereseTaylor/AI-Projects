
# 🧠 Machine Learning Projects

Welcome to this collection of machine learning and recommender system projects! This repository includes various projects that uses machine learning and deep learning models for a variety of applications, ranging from image classification to music recommendation and fraud detection.

---

## 📂 Projects Overview

1. **Fruit Classifier (CNN)**:
   - Uses a Convolutional Neural Network (CNN) to classify images of fruits as fresh or rotten.
   - **Key Features**: Deep learning with convolutional layers, data augmentation for generalization.
   - **Accuracy**: 97%
   - **File**: `CNN.ipynb`
   - **Dataset**: https://zenodo.org/records/7224690

2. **Music Recommender System**:
   - Implements various recommendation algorithms to suggest music tracks to users based on collaborative and content-based filtering.
   - **Key Features**: Collaborative filtering, content-based filtering, recommendation metrics.
   - **File**: `Recommenders.py`

3. **Forest Fire Detection (SVC & Random Forest Models)**:
   - Contains implementations of Support Vector Classifier (SVC) and Random Forest models to detect forest fires based on environmental data.
   - **Key Features**: Multi-model comparison, binary classification.
   - **Accuracy**: 89%
   - **Files**: `SCV & KNN Models.py`, `RandomForest.py`
   - **Dataset**: https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset

4. **Bitcoin Money Laundering Detection (Random Forest Model)**:
   - Uses a Random Forest model to classify Bitcoin transactions as licit or illicit based on risk scores and transaction data.
   - **Key Features**: Risk score calculation, binary classification, fraud detection.
   - **File**: `RF_Model.ipynb`
   - **Accuracy**: 98%
   - **Dataset**: https://www.kaggle.com/datasets/pablodejuanfidalgo/augmented-elliptic-data-set

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
