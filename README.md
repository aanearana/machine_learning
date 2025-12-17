# 🤖 Proyecto de Machine Learning para la Detección Temprana del Nivel de Ansiedad (MLP)

## 1. 🎯 Introducción y Contexto del Problema

La ansiedad es una condición de salud que, si no se detecta a tiempo, puede colapsar los servicios sanitarios y aumentar los costos debido a pruebas y tratamientos innecesarios para síntomas psicosomáticos.

La **identificación temprana y precisa** de la ansiedad tiene un impacto directo y significativo en la calidad de vida de los pacientes.

## 2. 🧠 El Modelo de Machine Learning (MLP)

El modelo de Machine Learning actúa como un **detector de patrones sofisticado**, analizando las respuestas de un cuestionario sencillo y basándose en miles de casos reales.

### 2.1. Variables de Entrada (X) Clave

Las variables con mayor peso o importancia en el análisis son:

* **Nivel de Estrés**
* **Horas de Sueño**
* **Sesiones de Terapia / Apoyo**
* **Otras variables:** Frecuencia cardíaca, consumo de sustancias, historial familiar, etc.

### 2.2. Valor y Aplicaciones Prácticas

El sistema genera una **alerta instantánea** (ej. "Riesgo de Ansiedad: 85%") que sirve como herramienta de apoyo para el profesional sanitario.

* **Mejora la Eficiencia:** Permite al personal sanitario enfocar los recursos limitados de salud mental en los pacientes que realmente los necesitan.
* **Mejora la Atención al Paciente:** Facilita la **intervención temprana**, ayudando a los pacientes a recibir apoyo psicológico antes de que su ansiedad se vuelva grave o crónica.

## 3. ⚙️ Metodología de ML

### 3.1. Preprocesamiento de Datos

* **Target (Y):** Clasificación Binaria a **2 clases** ("No tiene ansiedad" / "Sí tiene ansiedad").
* **Codificación:** Se utilizó **Label Encoding** y **One-Hot Encoding** para las variables categóricas.
* **Desequilibrio:** Se aplicó **SMOTE** al conjunto de entrenamiento para equilibrar la distribución de la clase objetivo.

## 4. 📊 Rendimiento del Modelo (Random Forest)

El algoritmo **Random Forest** fue el modelo seleccionado, logrando la precisión más alta de **$95.59\%$**.

### 4.1. Matriz de Confusión

La matriz de confusión muestra el rendimiento del modelo sobre el conjunto de prueba:

| Predicción | **No tiene ansiedad** | **Tiene ansiedad** |
| :---: | :---: | :---: |
| **Real: No tiene ansiedad** | **1660** (Aciertos / TN) | **24** (Fallos / Falsas Alarmas - FP) |
| **Real: Tiene ansiedad** | **124** (Fallos / Falsos Negativos - FN) | **1521** (Aciertos / TP) |


### 4.2. Impacto de las Métricas

* **Detección de Ansiedad (Sensibilidad):** El modelo logra identificar correctamente la ansiedad en el **$95\%$ de los casos reales**, traduciéndose en que **9 de cada 10 pacientes en riesgo son identificados a tiempo**.
* **Falsas Alarmas (FP):** La tasa de "falsas alarmas" es muy baja ($24$ fallos), asegurando la máxima eficiencia y evitando el desperdicio de tiempo del profesional sanitario.

***

**Desarrollado por:** Ane Arana