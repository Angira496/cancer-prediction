# ==============================
# Lung Cancer Prediction Project
# ==============================

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Load Dataset
# ------------------------------

df = pd.read_csv("survey lung cancer.csv")

print("First 5 rows:")
print(df.head())

# ------------------------------
# 2. Data Preprocessing
# ------------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Encode target column
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'YES':1, 'NO':0})

# Features and Target
X = df.drop(columns=['GENDER','AGE','LUNG_CANCER'])
y = df['LUNG_CANCER']

# ------------------------------
# 3. Train Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------
# 4. Model Training
# ------------------------------

log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=60)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ------------------------------
# 5. Model Evaluation
# ------------------------------

y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

acc_log = accuracy_score(y_test, y_pred_log)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\nLogistic Regression Accuracy:", acc_log)
print("Random Forest Accuracy:", acc_rf)

# ------------------------------
# 6. User Input Prediction
# ------------------------------

print("\nEnter Patient Details (0 = No, 1 = Yes)")

smoke = int(input("Smoking: "))
yellow_fingers = int(input("Yellow Fingers: "))
anxiety = int(input("Anxiety: "))
peer_pressure = int(input("Peer Pressure: "))
chronic_d = int(input("Chronic Disease: "))
fatigue = int(input("Fatigue: "))
allergy = int(input("Allergy: "))
wheezing = int(input("Wheezing: "))
alcohol = int(input("Alcohol Consuming: "))
coughing = int(input("Coughing: "))
shortness_breath = int(input("Shortness of Breath: "))
swallowing_difficulty = int(input("Swallowing Difficulty: "))
chest_pain = int(input("Chest Pain: "))

user_input = [[
    smoke, yellow_fingers, anxiety, peer_pressure,
    chronic_d, fatigue, allergy, wheezing,
    alcohol, coughing, shortness_breath,
    swallowing_difficulty, chest_pain
]]

# Predict using best model (Random Forest usually better)
prediction = rf_model.predict(user_input)

# ------------------------------
# 7. Result
# ------------------------------

if prediction[0] == 0:
    print("\nResult: No Lung Cancer Detected")
else:
    print("\nResult: High Risk of Lung Cancer")