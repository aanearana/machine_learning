import pandas as pd
#Escalado
from sklearn.preprocessing import StandardScaler
#Preprocesamiento
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
#-------
data = pd.read_csv('../data/raw/enhanced_anxiety_dataset.csv')
#-------
X = data.drop(columns=["Anxiety Level (1-10)"])
y = data["Anxiety Level (1-10)"]
#-------
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
# Reconstruir el DataFrame balanceado
data_resampled = X_resampled.copy()
data_resampled["Anxiety Level (1-10)"] = y_resampled
#-------
gender_dummies = pd.get_dummies(data["Gender"], prefix="Gender")
data = pd.concat([data.drop(columns=["Gender"]), gender_dummies], axis=1)
data = pd.get_dummies(data, columns=['Occupation'])
#-------
bool_cols = data.select_dtypes(include=['bool']).columns
data[bool_cols] = data[bool_cols].astype(int)
#------
le = LabelEncoder()
cols_to_label_encode = ['Smoking', 'Family History of Anxiety', 'Dizziness', 'Medication', 'Recent Major Life Event']
for col in cols_to_label_encode:
    data[col] = le.fit_transform(data[col])
#-------
X_final = data.drop(columns=["Anxiety Level (1-10)"])
y_final = data["Anxiety Level (1-10)"]
#-------
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_final)
X_scaled = pd.DataFrame(
    X_scaled_array,
    columns=X_final.columns,
    index=X_final.index)
#-------
data_processed = pd.concat([X_scaled, y_final], axis=1)
#-------
train, test = train_test_split(data_processed, test_size=0.1, random_state=42)

train.to_csv('../data/train_test/anxiety_train.csv', index=False)
test.to_csv('../data/train_test/anxiety_test.csv', index=False)