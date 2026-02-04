# Diabetes Prediction Using Machine Learning and Deep Learning

This project focuses on building a predictive system for identifying diabetes risk using Machine Learning and Deep Learning techniques. The goal is to compare traditional machine learning models with a neural network–based approach and evaluate their effectiveness in medical diagnosis support.

---

## Problem Statement
Diabetes is a chronic disease with severe long-term health implications. Early prediction using medical data can support timely intervention and informed decision-making. This project aims to develop data-driven models that accurately classify diabetic and non-diabetic cases.

---

## Dataset
- Publicly available diabetes dataset  
- Contains medical attributes such as glucose level, blood pressure, BMI, age, etc.  
- Target variable: `Outcome`  
  - `0` → Non-diabetic  
  - `1` → Diabetic  

Dataset file:
data/diabetes.csv

---

## Methodology
1. Data loading and exploration  
2. Data cleaning and handling missing values  
3. Feature scaling using StandardScaler  
4. Feature selection using SelectKBest with ANOVA F-test  
5. Model training and evaluation  
6. Performance comparison  

---

## Algorithms Used

### Machine Learning
- Logistic Regression  
- Random Forest Classifier  

### Deep Learning
- Artificial Neural Network (ANN)  
  - Framework: TensorFlow / Keras  
  - Dense layers with ReLU activation  
  - Sigmoid activation for binary classification  

---

## Evaluation Metrics
- Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-score  

---

## Results Summary
- Logistic Regression provided a reliable baseline performance  
- Random Forest improved classification accuracy by capturing non-linear patterns  
- The Deep Learning model demonstrated strong predictive capability, particularly in identifying diabetic cases  
- Ensemble and neural network approaches outperformed baseline methods  

---

## Project Structure
diabetes-prediction/
│── README.md  
│── data/  
│   └── diabetes.csv  
│── notebooks/  
│   └── diabetes_prediction.ipynb  
│── src/  
│   └── model_training.py  
│── presentation/  
│   └── diabetes_prediction_presentation.pdf  

---

## How to Run
1. Clone the repository  
2. Install required libraries:
pip install pandas numpy scikit-learn tensorflow
3. Run the Jupyter Notebook or Python script  

---

## Author
Nion Rahaman Akash  
Background: Mathematics & Data Science  
Focus: Machine Learning, Deep Learning, Statistical Modeling  

---

## Note
This project was completed as part of academic coursework and is intended for educational and research demonstration purposes.
