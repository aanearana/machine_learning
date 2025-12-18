# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- CONSTANTES DE CONFIGURACIÓN ---
# Ajustar la ruta si el archivo 'final_model.pkl' no está en ../models respecto a app_streamlit
# Por ejemplo, si ejecutas 'streamlit run app_streamlit/app.py' desde el directorio padre
MODELS_DIR = '../models' 
MODEL_FILENAME = 'final_model.pkl'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)


# --- NOMBRES DE COLUMNAS DEL CUESTIONARIO A LOS DEL CSV DE ENTRENAMIENTO ---
# Este mapeo es para las columnas que no son OHE.
QUESTIONNAIRE_TO_MODEL_MAP = {
    'Edad': 'Age',
    'Horas de sueño': 'Sleep Hours', 
    'Horas de ejercicio a la semana': 'Physical Activity (hrs/week)',
    'Cantidad de cafeína diaria (mg)': 'Caffeine Intake (mg/day)',
    'Cantidad de alcohol a la semana': 'Alcohol Consumption (drinks/week)',
    'Fumador (Si/No)': 'Smoking', 
    'Antecedentes': 'Family History of Anxiety',
    'Nivel de estrés del 1 al 10': 'Stress Level (1-10)',
    'Frecuencia cardíaca': 'Heart Rate (bpm)',
    'Frecuencia respiratoria (respiraciones/min)': 'Breathing Rate (breaths/min)', 
    'Nivel de sudoración del 1 al 5': 'Sweating Level (1-5)',
    'Mareos (Si/No)': 'Dizziness', 
    'Medicación (Si/No)': 'Medication', 
    'Sesiones de terapia al mes': 'Therapy Sessions (per month)',
    'Acontecimiento importante reciente en la vida (Si/No)': 'Recent Major Life Event', 
    'Calidad de la dieta del 1 al 10': 'Diet Quality (1-10)',
    # Categóricas
    'Genero': 'Gender',
    'Puesto de trabajo': 'Occupation'
}

# Grupos de ocupación que el modelo espera después del OHE
OCCUPATION_GROUPS = [
    'Artist', 'Athlete', 'Chef', 'Doctor', 'Engineer', 'Freelancer', 
    'Lawyer', 'Musician', 'Nurse', 'Other', 'Scientist', 'Student', 'Teacher'
]

# --- LISTA DE CARACTERÍSTICAS ESPERADAS (32 COLUMNAS) ---
# Copiada del script de evaluación para asegurar el orden correcto
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

@st.cache_resource # Almacena en caché el modelo para no recargarlo
def load_model(path):
    """Carga el modelo serializado (PKL)."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Archivo de modelo no encontrado en {path}.")
        st.stop() # Detiene la ejecución si el modelo no está
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()


def preprocess_data(raw_data: dict, feature_names_order: list):
    # 1. Crear DataFrame
    df_new_user = pd.DataFrame([raw_data]) 

    # 2. Mapeo Binario (Si/No a 1/0)
    si_no_cols = ['Fumador (Si/No)', 'Antecedentes', 'Mareos (Si/No)', 'Medicación (Si/No)', 'Acontecimiento importante reciente en la vida (Si/No)']
    for col in si_no_cols:
        # Mapeo: SI -> 1, NO -> 0
        df_new_user[col] = df_new_user[col].map({'Sí': 1, 'No': 0, 'SI': 1, 'NO': 0})
        # Si el usuario no ingresó nada, usamos 0 como default (o el que consideres mejor)
        df_new_user[col] = pd.to_numeric(df_new_user[col], errors='coerce').fillna(0).astype(int) 
    
    # 3. Conversión de campos numéricos (ya debería serlo por los widgets de Streamlit)
    numeric_q_cols = [
        'Edad', 'Horas de sueño', 'Horas de ejercicio a la semana', 
        'Cantidad de cafeína diaria (mg)', 'Cantidad de alcohol a la semana', 
        'Nivel de estrés del 1 al 10', 'Frecuencia cardíaca', 
        'Frecuencia respiratoria (respiraciones/min)', 'Nivel de sudoración del 1 al 5', 
        'Sesiones de terapia al mes', 'Calidad de la dieta del 1 al 10'
    ]
    for col in numeric_q_cols:
        df_new_user[col] = pd.to_numeric(df_new_user[col], errors='coerce')


    # 4. Mapeo de Nombres de Columnas y Alineación
    
    # Inicializar el DataFrame con todas las columnas esperadas por el modelo a 0
    X_predict_aligned = pd.DataFrame(0, index=[0], columns=feature_names_order)
    
    # Transferir los datos numéricos y binarios
    for q_col, model_col in QUESTIONNAIRE_TO_MODEL_MAP.items():
        if q_col not in ['Genero', 'Puesto de trabajo'] and model_col in feature_names_order:
            X_predict_aligned[model_col] = df_new_user[q_col].iloc[0]
            
    # --- Manejo de Variables Categóricas (OHE) ---
    
    # Género: 
    gender_input = raw_data['Genero']
    gender_column_name = f'Gender_{gender_input}' 
    
    if gender_column_name in feature_names_order:
        X_predict_aligned[gender_column_name] = 1
    
    # Puesto de trabajo (OHE Específico):
    job_input = raw_data['Puesto de trabajo']
    is_job_matched = False
    
    for group in OCCUPATION_GROUPS:
        expected_column_name = f'Occupation_{group}'
        
        if job_input == group and expected_column_name in feature_names_order:
            X_predict_aligned[expected_column_name] = 1
            is_job_matched = True
            break 
            
    # Si la ocupación no coincide con ninguna conocida, usamos 'Occupation_Other'
    if not is_job_matched:
        other_column = 'Occupation_Other'
        if other_column in feature_names_order:
            X_predict_aligned[other_column] = 1

    # Final check: Aseguramos que TODAS las columnas estén en el orden correcto
    X_final = X_predict_aligned[feature_names_order]
    
    return X_final


# --- CONFIGURACIÓN DE STREAMLIT ---

# CSS para centrar y estilizar el botón
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        display: block;
        margin: auto;              /* centra horizontalmente */
        background-color: #1e0ac6; /* color azul */
        color: white;              /* texto blanco */
        font-weight: bold;         /* negrita */
        font-size: 18px;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Predicción de Ansiedad", layout="centered")

st.title("Predictor de Riesgo de Ansiedad")

# 1. Cargar el Modelo
rf_model = load_model(MODEL_PATH)

# Obtener las características esperadas (el ORDEN es VITAL)
try:
    FEATURES_ORDER = rf_model.feature_names_ 
except AttributeError:
    # Solución de emergencia
    FEATURES_ORDER = FEATURES_ORDER_MANUAL


# --- FORMULARIO DE ENTRADA DE DATOS ---
# --- CSS para estilizar el botón ---
# --- FORMULARIO ---
with st.form("ansiedad_form"):
    # Distribución en columnas para mejor visualización
    col1, col2 = st.columns(2)

    with col1:
        edad = st.slider("Edad", 18, 80, 30)
        genero = st.selectbox("Género", ['Male', 'Female', 'Other'])
        puesto_trabajo = st.selectbox("Puesto de trabajo", OCCUPATION_GROUPS)
        st.markdown("####")
        horas_sueno = st.slider("Horas de sueño (promedio)", 0.0, 12.0, 7.0, 0.5)
        horas_ejercicio = st.slider("Horas de ejercicio a la semana", 0.0, 20.0, 3.0, 0.5)
        calidad_dieta = st.slider("Calidad de la dieta (1=Mala, 10=Excelente)", 1, 10, 5)
        cafeina = st.number_input("Cantidad de cafeína diaria (mg)", min_value=0, max_value=1000, value=100)
        alcohol = st.number_input("Cantidad de alcohol a la semana (unidades)", min_value=0, max_value=50, value=0)
        evento_reciente = st.selectbox("Acontecimiento importante reciente en la vida", ['No', 'Sí'])

    with col2:
        stress_level = st.slider("Nivel de estrés (1-10)", 1, 10, 5)
        frecuencia_cardiaca = st.number_input("Frecuencia cardíaca (latidos/min)", min_value=40, max_value=200, value=75)
        frecuencia_respiratoria = st.number_input("Frecuencia respiratoria (respiraciones/min)", min_value=5, max_value=40, value=15)
        sudoracion = st.slider("Nivel de sudoración (1-5)", 1, 5, 3)
        st.markdown("####")
        fumador = st.selectbox("Fumador", ['No', 'Sí'])
        antecedentes = st.selectbox("Antecedentes Familiares de Ansiedad", ['No', 'Sí'])
        mareos = st.selectbox("Mareos", ['No', 'Sí'])
        medicacion = st.selectbox("Medicación Psiquiátrica", ['No', 'Sí'])
        terapia = st.number_input("Sesiones de terapia al mes", min_value=0, max_value=10, value=0)

    # Centrar el botón usando columnas
    col_empty, col_button, col_empty2 = st.columns([2,1,2])
    with col_button:
        submitted = st.form_submit_button("Predecir")

if submitted:
    # 2. Recopilar y Pre-procesar los datos del usuario
    raw_user_data = {
        'Edad': edad,
        'Genero': genero,
        'Puesto de trabajo': puesto_trabajo,
        'Horas de sueño': horas_sueno,
        'Horas de ejercicio a la semana': horas_ejercicio,
        'Cantidad de cafeína diaria (mg)': cafeina,
        'Cantidad de alcohol a la semana': alcohol,
        'Fumador (Si/No)': fumador,
        'Antecedentes': antecedentes,
        'Nivel de estrés del 1 al 10': stress_level,
        'Frecuencia cardíaca': frecuencia_cardiaca,
        'Frecuencia respiratoria (respiraciones/min)': frecuencia_respiratoria,
        'Nivel de sudoración del 1 al 5': sudoracion,
        'Mareos (Si/No)': mareos,
        'Medicación (Si/No)': medicacion,
        'Sesiones de terapia al mes': terapia,
        'Acontecimiento importante reciente en la vida (Si/No)': evento_reciente,
        'Calidad de la dieta del 1 al 10': calidad_dieta,
    }
    
    try:
        X_predict_final = preprocess_data(raw_user_data, FEATURES_ORDER)
        
        # 3. Generar Predicción
        y_pred = rf_model.predict(X_predict_final)
        y_proba = rf_model.predict_proba(X_predict_final)[:, 1] # Probabilidad de clase 1 (Ansiedad)
        
        prediction_value = y_pred[0]
        prob_ansiedad = y_proba[0]
        
        # Mapeo de resultados
        # Mapeo de resultados usando la probabilidad
        umbral = 0.4  
        if prob_ansiedad > umbral:
            st.error(f"Resultado: **PRESENCIA DE ANSIEDAD**")
        else:
            st.success(f"Resultado: **AUSENCIA DE ANSIEDAD**")

        st.markdown("---")

        
    except Exception as e:
        st.error(f" Error al ejecutar la predicción: {e}")

#streamlit run 1_Streamlit.py 