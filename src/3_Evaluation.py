import joblib
import pandas as pd
import numpy as np
import os

# CONSTANTES DE CONFIGURACIÓN
MODELS_DIR = '../models' # Ajusta si tu ruta es diferente
MODEL_FILENAME = 'final_model.pkl'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
TARGET_COLUMN_PREDICT = 'Anxiety_Group' 

# NOMBRES DE COLUMNAS DEL CUESTIONARIO A LOS DEL CSV DE ENTRENAMIENTO ---
# Este mapeo es para las columnas que no son OHE.
QUESTIONNAIRE_TO_MODEL_MAP = {
    'Edad': 'Age',
    'Horas de sueño': 'Sleep Hours', 
    'Horas de ejercicio a la semana': 'Physical Activity (hrs/week)',
    'Cantidad de cafeína diaria (mg)': 'Caffeine Intake (mg/day)',
    'Cantidad de alcohol a la semana': 'Alcohol Consumption (drinks/week)',
    'Fumador (Si/No)': 'Smoking', 
    'Antecedentes': 'Family History of Anxiety', # Nombre ajustado al CSV
    'Nivel de estrés del 1 al 10': 'Stress Level (1-10)',
    'Frecuencia cardíaca': 'Heart Rate (bpm)',
    'Frecuencia respiratoria (respiraciones/min)': 'Breathing Rate (breaths/min)', # Nombre ajustado al CSV
    'Nivel de sudoración del 1 al 5': 'Sweating Level (1-5)',
    'Mareos (Si/No)': 'Dizziness', 
    'Medicación (Si/No)': 'Medication', 
    'Sesiones de terapia al mes': 'Therapy Sessions (per month)',
    'Acontecimiento importante reciente en la vida (Si/No)': 'Recent Major Life Event', 
    'Calidad de la dieta del 1 al 10': 'Diet Quality (1-10)',
    # Categóricas (se manejan por separado para OHE)
    'Genero': 'Gender',
    'Puesto de trabajo': 'Occupation'
}

# Grupos de ocupación que el modelo espera después del OHE (incluyendo 'Other')
OCCUPATION_GROUPS = [
    'Artist', 'Athlete', 'Chef', 'Doctor', 'Engineer', 'Freelancer', 
    'Lawyer', 'Musician', 'Nurse', 'Other', 'Scientist', 'Student', 'Teacher'
]

#LISTA DE CARACTERÍSTICAS ESPERADAS (32 COLUMNAS) ---
# ESTA LISTA ES LA SOLUCIÓN AL ERROR DE 'feature_names_'.
# Debe ser la lista EXACTA y en el orden CORRECTO que usaste para entrenar X_train.
FEATURES_ORDER_MANUAL = [
    'Age', 'Sleep Hours', 'Physical Activity (hrs/week)', 'Caffeine Intake (mg/day)', 
    'Alcohol Consumption (drinks/week)', 'Smoking', 'Family History of Anxiety', 
    'Stress Level (1-10)', 'Heart Rate (bpm)', 'Breathing Rate (breaths/min)', 
    'Sweating Level (1-5)', 'Dizziness', 'Medication', 'Therapy Sessions (per month)', 
    'Recent Major Life Event', 'Diet Quality (1-10)', 
    'Gender_Female', 'Gender_Male', 'Gender_Other', 
    'Occupation_Artist', 'Occupation_Athlete', 'Occupation_Chef', 'Occupation_Doctor', 
    'Occupation_Engineer', 'Occupation_Freelancer', 'Occupation_Lawyer', 
    'Occupation_Musician', 'Occupation_Nurse', 'Occupation_Other', 
    'Occupation_Scientist', 'Occupation_Student', 'Occupation_Teacher'
]


def get_user_data_and_preprocess(feature_names_order):
    """
    Recoge las respuestas del cuestionario y las procesa (mapea y OHE) 
    al formato exacto que espera el modelo.
    """
    print("Cuestionario de Predicción de Ansiedad")
    print("Por favor, introduce los siguientes datos:")
    
    # Recolección de Datos (Simulación de Formulario)
    raw_data = {}
    
    # Normalizamos la entrada de Género y Ocupación para OHE
    raw_data['Edad'] = input("Edad: ")
    raw_data['Genero'] = input("Género (Male/Female/Other): ").strip().title() 
    raw_data['Puesto de trabajo'] = input(f"Puesto de trabajo ({'/'.join(OCCUPATION_GROUPS)}): ").strip().title()
    
    raw_data['Horas de sueño'] = input("Horas de sueño: ")
    raw_data['Horas de ejercicio a la semana'] = input("Horas de ejercicio a la semana: ")
    raw_data['Cantidad de cafeína diaria (mg)'] = input("Cantidad de cafeína diaria (mg): ")
    raw_data['Cantidad de alcohol a la semana'] = input("Cantidad de alcohol a la semana (unidades): ")
    
    raw_data['Fumador (Si/No)'] = input("Fumador (Si/No): ").upper().strip()
    raw_data['Antecedentes'] = input("Antecedentes Familiares de Ansiedad (Si/No): ").upper().strip()
    raw_data['Nivel de estrés del 1 al 10'] = input("Nivel de estrés (1-10): ")
    raw_data['Frecuencia cardíaca'] = input("Frecuencia cardíaca (latidos/min): ")
    raw_data['Frecuencia respiratoria (respiraciones/min)'] = input("Frecuencia respiratoria (respiraciones/min): ")
    raw_data['Nivel de sudoración del 1 al 5'] = input("Nivel de sudoración (1-5): ")
    raw_data['Mareos (Si/No)'] = input("Mareos (Si/No): ").upper().strip()
    raw_data['Medicación (Si/No)'] = input("Medicación Psiquiátrica (Si/No): ").upper().strip()
    raw_data['Sesiones de terapia al mes'] = input("Sesiones de terapia al mes: ")
    raw_data['Acontecimiento importante reciente en la vida (Si/No)'] = input("Acontecimiento importante reciente en la vida (Si/No): ").upper().strip()
    raw_data['Calidad de la dieta del 1 al 10'] = input("Calidad de la dieta (1-10): ")
    
    df_new_user = pd.DataFrame([raw_data]) 

    # Mapeo Binario (Si/No a 1/0) y Numérico
    si_no_cols = ['Fumador (Si/No)', 'Antecedentes', 'Mareos (Si/No)', 'Medicación (Si/No)', 'Acontecimiento importante reciente en la vida (Si/No)']
    for col in si_no_cols:
        # Mapeo: SI -> 1, NO -> 0
        df_new_user.loc[df_new_user[col] == 'SI', col] = 1
        df_new_user.loc[df_new_user[col] == 'NO', col] = 0
        df_new_user[col] = pd.to_numeric(df_new_user[col], errors='coerce').fillna(0).astype(int) 
    
    # Conversión de campos numéricos 
    numeric_q_cols = [
        'Edad', 'Horas de sueño', 'Horas de ejercicio a la semana', 
        'Cantidad de cafeína diaria (mg)', 'Cantidad de alcohol a la semana', 
        'Nivel de estrés del 1 al 10', 'Frecuencia cardíaca', 
        'Frecuencia respiratoria (respiraciones/min)', 'Nivel de sudoración del 1 al 5', 
        'Sesiones de terapia al mes', 'Calidad de la dieta del 1 al 10'
    ]
    for col in numeric_q_cols:
        df_new_user[col] = pd.to_numeric(df_new_user[col], errors='coerce')


    # Mapeo de Nombres de Columnas y Alineación
    
    # Inicializar el DataFrame con todas las columnas esperadas por el modelo a 0
    X_predict_aligned = pd.DataFrame(0, index=[0], columns=feature_names_order)
    
    # Transferir los datos numéricos y binarios
    for q_col, model_col in QUESTIONNAIRE_TO_MODEL_MAP.items():
        if q_col not in ['Genero', 'Puesto de trabajo'] and model_col in feature_names_order:
            X_predict_aligned[model_col] = df_new_user[q_col].iloc[0]
            
    # Manejo de Variables Categóricas (One-Hot Encoding - OHE)
    
    # Género: 
    gender_input = df_new_user['Genero'].iloc[0] 
    gender_column_name = f'Gender_{gender_input}' 
    
    if gender_column_name in feature_names_order:
        X_predict_aligned[gender_column_name] = 1
    
    # Puesto de trabajo (OHE Específico):
    job_input = df_new_user['Puesto de trabajo'].iloc[0]
    is_job_matched = False
    
    for group in OCCUPATION_GROUPS:
        expected_column_name = f'Occupation_{group}'
        
        # Comprobamos si el input del usuario coincide (case-insensitive)
        if job_input.lower() == group.lower() and expected_column_name in feature_names_order:
            X_predict_aligned[expected_column_name] = 1
            is_job_matched = True
            break 
            
    # Si la ocupación no coincide con ninguna conocida, usamos 'Occupation_Other'
    if not is_job_matched:
        other_column = 'Occupation_Other'
        if other_column in feature_names_order:
            X_predict_aligned[other_column] = 1
            print(f"Aviso: La ocupación '{job_input}' fue clasificada como '{other_column}'.")


    # Final check: Aseguramos que TODAS las columnas estén en el orden correcto
    X_final = X_predict_aligned[feature_names_order]
    
    print("\nDatos de usuario procesados y alineados con el modelo.")
    return X_final


def load_model(path):
    """Carga el modelo serializado (PKL)."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        print(f"Error: Archivo de modelo no encontrado en {path}. ¡Asegúrate de que 'final_model.pkl' exista en el directorio '../models'!")
        exit()

# --- EJECUCIÓN PRINCIPAL ---

if __name__ == "__main__":
    
    print("Script de Predicción para Nuevo Usuario Iniciado")
    
    # 1. Cargar el Modelo
    rf_model = load_model(MODEL_PATH)
    print(f"Modelo '{MODEL_FILENAME}' cargado correctamente.")

    # 2. Obtener las características esperadas (el ORDEN es VITAL)
    FEATURES_ORDER = []
    try:
        # Intentamos obtener la lista del modelo (el método correcto)
        FEATURES_ORDER = rf_model.feature_names_ 
        print(f"Se esperan {len(FEATURES_ORDER)} características (obtenidas del modelo).")
    except AttributeError:
        # SOLUCIÓN DE EMERGENCIA: Usamos la lista de columnas codificada si falta el atributo
        FEATURES_ORDER = FEATURES_ORDER_MANUAL
        print(f"Error: El modelo PKL no contiene 'feature_names_'.")
        print(f"Usando la lista de {len(FEATURES_ORDER)} características definida manualmente.")


    # 3. Recoger y Pre-procesar los datos del usuario
    X_predict_final = get_user_data_and_preprocess(FEATURES_ORDER)
    
    # 4. Generar Predicción
    try:
        print("\nGenerando Predicción")
        y_pred = rf_model.predict(X_predict_final)
        
        # Mapeo binario (0 o 1)
        prediction_map = {
            1: "PRESENCIA DE ANSIEDAD", 
            0: "AUSENCIA DE ANSIEDAD"
        }
        
        prediction_value = y_pred[0]
        resultado = prediction_map.get(prediction_value, f"Valor de predicción desconocido: {prediction_value}")
        
        print(f"\n¡PREDICCIÓN FINAL!")
        print(f"El resultado para este usuario es: **{resultado}**")
        
    except Exception as e:
        print(f"Error: {e}")
        exit()