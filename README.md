# Diabetes Prediction Using Machine Learning and Deep Learning

This project focuses on building a predictive system for identifying diabetes risk using Machine Learning and Deep Learning techniques. The goal is to compare traditional machine learning models with a neural network-based approach and evaluate their effectiveness in medical diagnosis support.

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
1. Data loading and exploration (null-value check, feature/target split)  
2. Train/test split: 80% training (614 samples), 20% test (154 samples)  
3. Feature scaling using StandardScaler, fitted on the training set only to avoid data leakage  
4. Model training: Logistic Regression, Random Forest, and an Artificial Neural Network  
5. Evaluation with confusion matrices and class-wise precision, recall, and F1-score  
6. Performance comparison and analysis of the neural network's training curves  

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

Test set performance (154 samples, Class 0 = Non-diabetic, Class 1 = Diabetic):

| Model | Accuracy | Precision (0 / 1) | Recall (0 / 1) | F1-score (0 / 1) |
|---|---|---|---|---|
| Logistic Regression | 75% | 0.81 / 0.65 | 0.80 / 0.67 | 0.81 / 0.66 |
| Random Forest | 73% | 0.79 / 0.63 | 0.80 / 0.62 | 0.79 / 0.62 |
| ANN (Deep Learning) | 75% | 0.81 / 0.65 | 0.81 / 0.65 | 0.81 / 0.65 |

- Logistic Regression and the ANN achieved the best accuracy (75%), performing almost identically across precision, recall, and F1-score.
- Random Forest slightly underperformed the other two models (73% accuracy) on this dataset.
- All three models identify non-diabetic cases more reliably than diabetic ones, reflecting the class imbalance in the dataset. Future work could explore class balancing (e.g. SMOTE), threshold tuning, or additional features to improve recall on diabetic cases.
- The neural network did not outperform the simpler Logistic Regression. Its training accuracy kept rising (to roughly 85–89%) while validation accuracy plateaued around 72% and validation loss increased after the first few epochs, a sign of overfitting on a small dataset (614 training samples).

---

## Limitations and Next Steps
- **Hidden missing values:** the dataset reports no null values, but several features use `0` to mean "not recorded" (Insulin 48.7% of rows, SkinThickness 29.6%, BloodPressure 4.6%, BMI 1.4%, Glucose 0.7%). These zeros are currently used as real values; imputing them is the next planned improvement.
- **Single train/test split:** results come from one 80/20 split; stratified k-fold cross-validation would give more reliable estimates.
- **Reproducibility:** random seeds are not set for the Random Forest and neural network, so their scores can vary slightly between runs.
- **Diabetic recall:** class weighting, decision-threshold tuning, and ROC-AUC reporting are planned to better capture the minority (diabetic) class.

---

## Project Structure
```
diabetes-prediction/
├── README.md
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── diabetes_prediction.ipynb
├── src/
│   └── model_training.py
└── presentation/
    └── diabetes_prediction_presentation.pdf
```

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
