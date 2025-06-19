# MachineLearningProjectPredicción del Tiempo con Machine Learning
Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje automático para la predicción del tiempo utilizando un conjunto de datos meteorológicos multiciudad. El dataset contiene observaciones diarias de numerosas variables relacionadas con el clima (cobertura nubosa, radiación solar, precipitaciones, temperaturas, entre otros) recogidas en distintas ciudades de Europa como Basel, Budapest, De Bilt, Dresden, Heathrow, Roma, entre otras.

📁 Datos
El conjunto de datos incluye variables como:

Temperatura media, mínima y máxima

Humedad relativa

Presión atmosférica

Radiación solar global

Horas de sol

Cobertura nubosa

Velocidad del viento y ráfagas

Precipitaciones

Cada observación está identificada por una fecha (DATE) y el mes (MONTH), y contiene las variables anteriores para múltiples estaciones meteorológicas europeas.

🧠 Objetivo
Desarrollar modelos predictivos capaces de anticipar variables meteorológicas clave (como temperatura, precipitación o radiación solar) en función de los datos históricos multivariable y multiciudad. Este proyecto también servirá como base para integrar futuras mejoras, como:

Predicción a varios días vista

Incorporación de series temporales externas (como fenómenos climáticos globales)

Interfaz visual interactiva para consulta del tiempo previsto

🛠️ Tecnologías previstas
Python

Pandas y NumPy

Scikit-learn

XGBoost / LightGBM

Visualización con Matplotlib / Seaborn

(en el futuro: dashboard con Streamlit o Plotly Dash)

🚀 Próximos pasos
Análisis exploratorio de datos (EDA)

Preparación y limpieza del dataset

Selección de variables relevantes

Entrenamiento de modelos regresivos

Evaluación del rendimiento

Desarrollo de la interfaz de usuario para predicción visual
