import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

BOOLEAN_COLS = ['Smoking', 'Family History of Anxiety', 'Dizziness', 'Medication', 'Recent Major Life Event']
TARGET_COL = 'Anxiety_Group'
RANDOM_STATE = 42

def load_and_initial_process(data_path, test_path):
    data = pd.read_csv(data_path)
    test = pd.read_csv(test_path)

    bins = [0, 5, 10]
    labels = [0, 1]
    data[TARGET_COL] = pd.cut(data['Anxiety Level (1-10)'], bins=bins, labels=labels, right=True, include_lowest=True).astype(int)
    data = data.drop(columns='Anxiety Level (1-10)')

    data = pd.get_dummies(data, columns=["Gender", "Occupation"])
    test = pd.get_dummies(test, columns=["Gender", "Occupation"])
    
    data = data.astype({col: int for col in data.select_dtypes(include='bool').columns})
    test = test.astype({col: int for col in test.select_dtypes(include='bool').columns})

    return data, test

def encode_and_balance(data, test):
    for col in BOOLEAN_COLS:
        if col in data.columns and col in test.columns:
            le = LabelEncoder()
            le.fit(data[col])
            data[col] = le.transform(data[col])
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            test[col] = test[col].map(mapping).fillna(-1).astype(int) 
        else:
            print(f"La columna '{col}' no se encontró en ambos datasets.")

    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y) 
    data_resampled = X_res.copy()
    data_resampled[TARGET_COL] = y_res
    return data_resampled, test

def split_scale_and_save(data_resampled, test, test_size=0.1, random_state=RANDOM_STATE):
    train, test_split = train_test_split(data_resampled, test_size=test_size, random_state=random_state)
    
    X_train = train.drop(TARGET_COL, axis=1)
    y_train = train[TARGET_COL]
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_scaled[TARGET_COL] = y_train.reset_index(drop=True)

    test_split = test_split.drop(columns=[TARGET_COL])
    train_scaled.to_csv('../data/train_test/anxiety_train.csv', index=False)
    test_split.to_csv('../data/train_test/anxiety_test.csv', index=False) 
    
    print("Train set guardado en: '../data/train_test/anxiety_train.csv')")
    print("Test set guardado en: '../data/train_test/anxiety_test.csv')")
    return train_scaled, test_split

if __name__ == '__main__':
    DATA_PATH = '../data/raw/enhanced_anxiety_dataset.csv'
    TEST_PATH = '../data/raw/family_anxiety_14_dataset.csv'
    data_initial, test_initial = load_and_initial_process(DATA_PATH, TEST_PATH)
    data_resampled, test_processed = encode_and_balance(data_initial, test_initial)
    final_train, final_test_split = split_scale_and_save(data_resampled, test_processed)
    
