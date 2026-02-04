#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[2]:


# Load the dataset
df = pd.read_csv('../data/diabetes.csv')  # Update this path if necessary

# Check the first few rows
df.head()


# In[3]:


# Check for missing values
print(df.isnull().sum())

# Split data into features and target
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']                # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[4]:


# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Define the deep learning model using Input layer
dl_model = Sequential()

# Input layer
dl_model.add(Input(shape=(X_train.shape[1],)))  # Specify input shape

# Hidden layers
dl_model.add(Dense(32, activation='relu'))
dl_model.add(Dense(16, activation='relu'))

# Output layer
dl_model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
dl_model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)


# In[6]:


# Predictions with Logistic Regression
lr_predictions = lr_model.predict(X_test)
print("Logistic Regression Evaluation:")
print(confusion_matrix(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

# Predictions with Random Forest
rf_predictions = rf_model.predict(X_test)
print("Random Forest Evaluation:")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

# Predictions with Deep Learning Model
dl_predictions = (dl_model.predict(X_test) > 0.5).astype("int32")
print("Deep Learning Model Evaluation:")
print(confusion_matrix(y_test, dl_predictions))
print(classification_report(y_test, dl_predictions))


# In[ ]:




