🌦️ Weather Forecast App — Predicción Climática Inteligente
Weather Forecast App es una aplicación interactiva desarrollada con Streamlit que permite predecir el clima futuro de una ciudad a partir de una fecha seleccionada. Utiliza modelos de aprendizaje automático entrenados con datos históricos para estimar:

🌡️ Temperatura media, mínima y máxima

🌧️ Probabilidad de precipitación

🧠 ¿Cómo funciona?
A partir de una fecha introducida por el usuario, la app:

Extrae información temporal relevante (como el mes, el día del año y el día de la semana)

Utiliza modelos de Machine Learning para predecir variables meteorológicas clave

Muestra los resultados de forma clara y visualmente atractiva

También en base a las variables meterorológicas introducidas por el usuario, la aplicación devuelve al estación meteorologica en la que se encuentra la ciudad seleccionada.

🤖 Modelos utilizados
Random Forest Regressor: para predecir variables continuas como temperatura y precipitación

K-Nearest Neighbors (KNN): para tareas complementarias de clasificación o comparación

Random Forest classifier, para predecir el tiempo que va a hacer el día seleccionado por el usuario.

Los modelos han sido entrenados con datos históricos meteorológicos de una ciudad europea.

🛠️ Tecnologías empleadas
Python

Streamlit

Scikit-learn

Pandas & NumPy

CSS personalizado para una interfaz moderna

▶️ Cómo ejecutar la app
bash
# Clona el repositorio
git clone https://github.com/oriolaguirre/MachineLearningProject.git
cd MachineLearningProject

# Instala las dependencias
pip install -r requirements.txt

# Ejecuta la app
streamlit run Front_weather.py
✨ Características destacadas
Interfaz intuitiva y responsive

Fondo visual personalizado

Inputs estilizados para mejor legibilidad

Predicción meteorológica basada en fecha futura

📅 Próximas mejoras
Visualización de predicciones en gráficos

Clasificación de imagenes satelitales para saber el tiempo que hará.

Integración con APIs meteorológicas en tiempo real
