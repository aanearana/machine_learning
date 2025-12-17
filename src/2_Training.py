import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
import os

DATA_PATH = '../data/train_test/anxiety_train.csv'
MODELS_DIR = '../models'
RANDOM_STATE = 55
TEST_SIZE = 0.15

PARAM_GRID = {
    'n_estimators': [300],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

def perform_grid_search(data_path, param_grid, random_state, test_size):
    data = pd.read_csv(data_path)

    X = data.drop(columns='Anxiety_Group')
    y = data['Anxiety_Group']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state), # Asegurar reproducibilidad
        param_grid=param_grid,
        cv=5,
        scoring='balanced_accuracy',  
        n_jobs=-1,
        verbose=2)
    
    grid.fit(X_train, y_train)
    pred = grid.predict(X_test) 
    
    print(f"Mejor CV Score: {grid.best_score_:.4f}")
    
    return grid

def save_model_and_config(grid_result, models_dir): 
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"   Directorio creado: {models_dir}")

    final_model_path = os.path.join(models_dir, 'final_model.pkl')
    joblib.dump(grid_result.best_estimator_, final_model_path) 
    print(f"   Modelo final guardado en: {final_model_path}")

    config_path = os.path.join(models_dir, 'model_config.yaml')
    model_config = {
        'model_type': 'RandomForestClassifier',
        'best_score_cv': float(grid_result.best_score_), 
        'hyperparameters': grid_result.best_params_}

    with open(config_path, 'w') as file:
        yaml.dump(model_config, file, default_flow_style=False)
    print(f"Configuración del modelo guardada en: {config_path}")


if __name__ == '__main__':
    try:
        best_grid = perform_grid_search(
            data_path=DATA_PATH,
            param_grid=PARAM_GRID,
            random_state=RANDOM_STATE,
            test_size=TEST_SIZE)

        save_model_and_config(best_grid, MODELS_DIR)
    except FileNotFoundError:
        print(f"ERROR: Archivo de datos no encontrado en {DATA_PATH}. Asegúrate de que el preprocesamiento se ejecutó.")
    except Exception as e:
        print(f"ERROR inesperado durante el entrenamiento: {e}")