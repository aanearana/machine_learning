import pandas as pd

#Modelo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Metricas
from sklearn.metrics import accuracy_score

#pickle
import pickle

data_train = pd.read_csv('../data/train_test/anxiety_train.csv')

X = data_train[[
        "Stress Level (1-10)",
        "Therapy Sessions (per month)",
        "Caffeine Intake (mg/day)",
        "Heart Rate (bpm)",
        "Breathing Rate (breaths/min)",
        "Family History of Anxiety",
        "Sweating Level (1-5)",
        "Alcohol Consumption (drinks/week)",
        "Dizziness",
        "Age",
        "Diet Quality (1-10)",
        "Physical Activity (hrs/week)",
        "Sleep Hours"
    ]]

y = data_train['Anxiety Level (1-10)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=55)

log_reg = RandomForestClassifier(n_estimators=50, random_state=55)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

print("Accuracy Random Forest:", acc_log)