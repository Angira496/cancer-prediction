# ==============================
# Lung Cancer Prediction Web App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("Lung Cancer Prediction System")
st.write("Machine Learning based lung cancer prediction with data insights")

# ------------------------------
# 1. Load Dataset
# ------------------------------

df = pd.read_csv("survey lung cancer.csv")

st.header("Dataset Preview")
st.write(df.head())

# ------------------------------
# 2. Data Preprocessing
# ------------------------------

df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'YES':1,'NO':0})
df['GENDER'] = df['GENDER'].replace({'M':1,'F':0})

X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# ------------------------------
# 3. Train Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.25,random_state=42
)

# ------------------------------
# 4. Model Training
# ------------------------------

log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=60)

log_model.fit(X_train,y_train)
rf_model.fit(X_train,y_train)

# ------------------------------
# 5. Model Evaluation
# ------------------------------

y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

acc_log = accuracy_score(y_test,y_pred_log)
acc_rf = accuracy_score(y_test,y_pred_rf)

st.header("Model Performance")

st.write("Logistic Regression Accuracy:",acc_log)
st.write("Random Forest Accuracy:",acc_rf)

# ------------------------------
# 6. Dataset Insights
# ------------------------------

st.header("Dataset Insights")

st.subheader("Missing Values")
st.write(df.isnull().sum())

# Cancer Distribution

st.subheader("Cancer Distribution")

fig1, ax1 = plt.subplots()

df['LUNG_CANCER'].value_counts().plot(kind='bar', ax=ax1)

ax1.set_title("Lung Cancer Cases")
ax1.set_xlabel("Cancer (0=No,1=Yes)")
ax1.set_ylabel("Count")

st.pyplot(fig1)

# Correlation Matrix

st.subheader("Feature Correlation")

corr = df.corr()

fig2, ax2 = plt.subplots()

cax = ax2.matshow(corr)
plt.colorbar(cax)

ax2.set_xticks(range(len(corr.columns)))
ax2.set_yticks(range(len(corr.columns)))

ax2.set_xticklabels(corr.columns, rotation=90)
ax2.set_yticklabels(corr.columns)

st.pyplot(fig2)

# Feature Importance

st.subheader("Feature Importance (Random Forest)")

importance = rf_model.feature_importances_
features = X.columns

fig3, ax3 = plt.subplots()

ax3.barh(features, importance)
ax3.set_title("Important Features for Lung Cancer Prediction")

st.pyplot(fig3)

# ------------------------------
# 7. Prediction Section
# ------------------------------

st.header("Predict Lung Cancer Risk")

age = st.number_input("Age",18,100)

gender = st.selectbox("Gender",[0,1])

smoke = st.selectbox("Smoking",[0,1])
yellow_fingers = st.selectbox("Yellow Fingers",[0,1])
anxiety = st.selectbox("Anxiety",[0,1])
peer_pressure = st.selectbox("Peer Pressure",[0,1])
chronic_d = st.selectbox("Chronic Disease",[0,1])
fatigue = st.selectbox("Fatigue",[0,1])
allergy = st.selectbox("Allergy",[0,1])
wheezing = st.selectbox("Wheezing",[0,1])
alcohol = st.selectbox("Alcohol Consuming",[0,1])
coughing = st.selectbox("Coughing",[0,1])
shortness_breath = st.selectbox("Shortness of Breath",[0,1])
swallowing_difficulty = st.selectbox("Swallowing Difficulty",[0,1])
chest_pain = st.selectbox("Chest Pain",[0,1])

if st.button("Predict"):

    user_input = pd.DataFrame([[

        gender,
        age,
        smoke,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_d,
        fatigue,
        allergy,
        wheezing,
        alcohol,
        coughing,
        shortness_breath,
        swallowing_difficulty,
        chest_pain

    ]],columns=X.columns)

    prediction = rf_model.predict(user_input)

    if prediction[0] == 0:
        st.success("No Lung Cancer Detected")
    else:
        st.error("High Risk of Lung Cancer")