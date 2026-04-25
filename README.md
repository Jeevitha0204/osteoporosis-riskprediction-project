# 🦴 Osteoporosis Detection using Multi-Modal Explainable AI

## 📌 Overview
This project presents a **multi-modal AI framework** for early detection of osteoporosis using both **knee X-ray images** and **clinical data**. The system combines deep learning and machine learning models to improve prediction accuracy and provide explainable results for better clinical understanding.

---

## 🚀 Key Features
- Binary Classification (Normal vs Osteoporosis)
- Multi-class Classification (Normal, Osteopenia, Osteoporosis)
- Clinical Data-based Risk Prediction
- Bone Risk Score Generation
- Explainable AI using Grad-CAM
- Web-based Deployment using Gradio

---

## 🧠 Models Used

### 🔹 Image-based Models (Deep Learning)
- ResNet50 ✅ (Best Performing)
- DenseNet121
- MobileNetV2
- EfficientNet

### 🔹 Clinical Data Model (Machine Learning)
- Gradient Boosting Classifier
- Random Forest
- Decision tree
- Logistic Regression

---

## ⚙️ Methodology

1. **Data Collection**
   - Combined 4 knee X-ray datasets from Kaggle
   - Used clinical dataset (CSV) with patient details

2. **Preprocessing**
   - Image resizing (224×224)
   - Normalization (0–255 → 0–1)
   - Data augmentation (flip, rotation, zoom)
   - Label standardization
   - Clinical data encoding & feature engineering

3. **Model Training**
   - Applied transfer learning on CNN models
   - Trained binary and multi-class classifiers
   - Trained Gradient Boosting for clinical data

4. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

5. **Explainability**
   - Grad-CAM used to highlight important regions in X-ray images

6. **Deployment**
   - Deployed as a web application using Hugging Face Spaces (Gradio)

---

## 📊 Results

- Binary Classification Accuracy: **94%**
- Multi-class Classification Accuracy: **83%**
- ResNet50 achieved best performance among all models

---

## 🌐 Deployment

The project is deployed as an interactive web application where users can:

- Upload X-ray images for prediction  
- Enter clinical data manually  
- Get prediction along with risk score  
- Visualize Grad-CAM heatmaps  

👉 *https://huggingface.co/spaces/jeevitha-app/Osteoporosis_risk_prediction*
<img width="1876" height="677" alt="image" src="https://github.com/user-attachments/assets/7e6a9c85-025f-43e1-bd04-4ff1593e785e" />
<img width="980" height="524" alt="image" src="https://github.com/user-attachments/assets/3c250083-8a14-4e04-aa87-d3bfebfb5657" />



---

## 📂 Dataset

- Knee X-ray datasets from Kaggle
  *https://www.kaggle.com/datasets/866059b7930a5c49cd77d94c1761840a19d88074cad74e8f0e0cfa2b236a6904*
  *https://www.kaggle.com/datasets/sachinkumar413/Osteoporosis-knee-dataset-preprocessed128x256*
  *https://www.kaggle.com/datasets/mrmann007/Osteoporosis*
  *https://www.kaggle.com/datasets/stevepython/Osteoporosis-knee-xray-dataset*
- Clinical dataset (CSV) with features like:
  - Age, Gender  
  - Calcium & Vitamin D intake  
  - Physical activity  
  - Medical history
  *https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis*

---

## ⚠️ Limitations

- No real-world clinical validation  
- Limited dataset size  
- Performance may vary across populations  

---

## 🔮 Future Work

- Clinical validation with real patient data  
- Integration with hospital systems  
- Mobile application development  
- Advanced explainable AI techniques  

---

## 💡 Conclusion

This project demonstrates an end-to-end **multi-modal AI system** that combines image and clinical data for improved osteoporosis detection. It provides both prediction and explainability, making it suitable as a clinical decision-support tool.

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- Scikit-learn  
- OpenCV  
- Gradio  
- Hugging Face Spaces  

---

## 👤 Author

**Jeevitha M**  
