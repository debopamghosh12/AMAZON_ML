# 🚀 Multimodal Product Pricing & Clickbait Detection System

  

This repository contains the solution for the **Smart Product Pricing Challenge** and the **Clickbait Headline Classification** task. The project leverages advanced machine learning techniques, including Multimodal Learning (Text + Image) and Natural Language Processing (NLP), to solve real-world e-commerce and media challenges.

  

## 📂 Project Overview

  

### **Challenge 1: Smart Product Pricing (Regression)**

**Goal:** Predict the price of a product based on its text description and image.

* **Dataset:** 75,000 training samples (Catalog text + Image URLs).

* **Approach:** Multimodal Learning.

    * **Text:** TF-IDF Vectorization & Regex-based feature engineering (Pack Quantity).

    * **Images:** Deep Learning feature extraction using a pre-trained **ResNet-50** model (2048-dimensional vectors).

    * **Model:** LightGBM Regressor (Gradient Boosting) trained on the combined feature set.

* **Key Technique:** Implemented a robust **chunking pipeline** to process 75k images and extract features without exceeding RAM limits on Google Colab.

  

### **Challenge 2: Clickbait Detection (Classification)**

**Goal:** Classify news headlines as "Clickbait" or "Not Clickbait".

* **Dataset:** 32,000 labeled headlines.

* **Approach:** NLP + Supervised Learning.

* **Model:** Logistic Regression with **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization.

* **Metric:** Optimized for F1-Score.

  

---

  

## 🛠️ Tech Stack

  

* **Language:** Python 3.10+

* **Core Libraries:**

    * `pandas`, `numpy` (Data Manipulation)

    * `scikit-learn` (ML Models, TF-IDF, Metrics)

    * `lightgbm` (High-performance Gradient Boosting)

* **Deep Learning:**

    * `torch` (PyTorch), `torchvision` (ResNet-50 Model)

* **Image Processing:**

    * `Pillow` (PIL), `requests` (Image Downloading)

* **Utilities:**

    * `tqdm` (Progress Bars), `joblib` (Model Saving), `scipy` (Sparse Matrices)

  

---

  

## ⚙️ How It Works (Methodology)

  

### **Pipeline 1: Product Price Prediction**

1.  **Data Ingestion (Chunking):** The 75k dataset is read in chunks of 2,500 rows to ensure memory efficiency.

2.  **Feature Engineering:**

    * *Text:* `TfidfVectorizer` (Unigrams + Bigrams, Top 20k features).

    * *Regex:* Extracted "Pack Quantity" from catalog descriptions.

    * *Image:* Downloaded images on-the-fly and passed them through **ResNet-50** (removing the final classification layer) to get deep visual features.

3.  **Storage:** Processed chunks (Text + Image features) were saved as sparse matrices (`.npz`) to Google Drive.

4.  **Training:** Loaded all chunks and trained a **LightGBM** model using Early Stopping to prevent overfitting.

  

### **Pipeline 2: Clickbait Classification**

1.  **Preprocessing:** Text cleaning (lowercase, special character removal).

2.  **Vectorization:** Converted headlines into sparse vectors using TF-IDF.

3.  **Training:** Trained a Logistic Regression classifier with class weighting to handle potential imbalances.

  

---

  

## 🚀 How to Run

  

### **Prerequisites**

* Google Colab (Recommended for GPU access)

* Google Drive (For persistent storage of large datasets)

  

### **Step 1: Setup**

Mount your Google Drive and ensure the datasets (`train.csv`, `test.csv`) are uploaded.

```python

from google.colab import drive

drive.mount('/content/drive')

  

### **Step 2: Run Feature Extraction (Challenge 1)**

Execute the chunking script to process images and text.

  

Input: train.csv

  

Output: processed_features/ (Folder containing .npz chunk files)

  

### **Step 3: Train Model**

Run the training script to load features and build the model.

  

Output: final_model_emergency.joblib

  

### **Step 4: Generate Submissions**

Run the prediction scripts for both challenges.

  

Price Prediction: Generates submission_final.csv

  

Clickbait: Generates submission_EMERGENCY_TEXT_ONLY.csv

  

📊 Results & Performance

Pricing Model: Leverages the power of visual data (ResNet features) combined with text descriptions to capture complex pricing patterns (e.g., visual quality, brand aesthetics).
