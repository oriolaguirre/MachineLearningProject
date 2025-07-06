import streamlit as st
import pandas as pd
import json
import numpy as np
import joblib

def set_custom_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://hydrosphere.co.uk/wp-content/uploads/2020/03/weather_predictions-e1584356162376.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
            text-shadow: 1px 1px 2px black;
        }

        .block-container {
            background-color: rgba(0, 0, 0, 0.5);
            padding: 2rem;
            border-radius: 10px;
        }

        .stButton>button {
            color: white;
            background-color: #1f77b4;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }

        .stSlider > div {
            color: white;
        }

       input, .stNumberInput input {
        background-color: rgba(0, 0, 0, 0.6) !important;
        color: white !important;
        border-radius: 5px;
        padding: 0.4rem;
        border: 1px solid rgba(255, 255, 255, 0.4);
        }

        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_background()


df = pd.read_csv('C:/Users/34666/Desktop/Oriol bootcamp the bridge/Curso/MachineLearningProject/MachineLearningProject/Data/raw/weather_prediction_dataset.csv')
# Cargar modelo y codificador entrenados
knn = joblib.load("../Models/modelo_knn.pkl")
le = joblib.load("../Models/label_encoder.pkl")
rndf = joblib.load("../Models/RandomForrest.pkl")
Regressor = joblib.load("../Models/modelo_regressor.pkl")

ciudades = sorted(set(col.split('_')[0] for col in df.columns if '_' in col and col != 'DATE' and col != 'MONTH'))

st.title("Prediccion Meteorologica")
ciudad_seleccionada = st.selectbox("Selecciona una ciudad", ciudades)

# Filtrar columnas que pertenecen a la ciudad seleccionada
columnas_ciudad = [col for col in df.columns if col.startswith(ciudad_seleccionada + "_") or col in ['DATE', 'MONTH']]
df_ciudad = df[columnas_ciudad]
df_ciudad.to_csv("../Data/processed/df_ciudad_filtrada.csv", index=False)

humedad = st.slider("Humedad (%)", 0.0, 100.0, 75.0)
presion = st.number_input("Presión (hPa)", value=1013.2)
radiacion = st.number_input("Radiación global", value=120.5)
precipitacion = st.number_input("Precipitación", value=0.0)
sol = st.number_input("Horas de sol", value=5.2)
temp_media = st.number_input("Temperatura media", value=15.3)
temp_min = st.number_input("Temperatura mínima", value=10.1)
temp_max = st.number_input("Temperatura máxima", value=20.4)

if st.button("Predecir estación"):
    punto_nuevo = np.array([[humedad, presion, radiacion, precipitacion, sol,
                         temp_media, temp_min, temp_max]])
    
    pred_num = knn.predict(punto_nuevo)[0]
    pred_num3 = rndf.predict(punto_nuevo)[0]

    estacionknn = le.inverse_transform([pred_num])[0]
    estacionrf = le.inverse_transform([pred_num3])[0]

    def estacion_a_mes(estacion):
        if estacion == 0:
            return "invierno"   # Enero
        elif estacion == 1:
         return "otoño"  # Abril
        elif estacion == 2:
                return "Primavera"  # Julio
        elif estacion == 3:
            return "verano" # Octubre

    estacionkn1 = estacion_a_mes(pred_num)
    estacionrf1 = estacion_a_mes(pred_num3)

    st.success(f"La estación del año en función de los parámetros es (Random Forrest): {estacionrf1}")
    st.success(f"La estación del año en función de los parámetros es (K neigh): {estacionkn1}")

# Mostrar resultado


k = st.slider("Selecciona el valor de k para KNN", min_value=1, max_value=20, value=5, step=1)


date = st.date_input("mete el dia",value=None)

if date:
    mes = date.month
    dia_del_ano = date.timetuple().tm_yday
    dia_semana = date.weekday()

st.write(f"Mes: {mes}, Día del año: {dia_del_ano}, Día de la semana: {dia_semana}")

if st.button("Predecir tiempo"):

    diaseleccionado = np.array([[mes, dia_del_ano, dia_semana]])
    pred_num = Regressor.predict(diaseleccionado)[0]
    st.subheader("🌤️ Predicción del clima")
    st.write(f"🌡️ Temperatura media: {pred_num[0]:.1f} °C")
    st.write(f"🌡️ Temperatura mínima: {pred_num[1]:.1f} °C")
    st.write(f"🌡️ Temperatura máxima: {pred_num[2]:.1f} °C")
    st.write(f"🌧️ Precipitación estimada: {pred_num[3]:.1f} mm")


with open("valor_k.json", "w") as f:
    json.dump({"k": k}, f)

with open("valor_k.json", "r") as f:
    data = json.load(f)

# Extraer los valores
k = data["k"]




import streamlit as st

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://hydrosphere.co.uk/wp-content/uploads/2020/03/weather_predictions-e1584356162376.jpg")


