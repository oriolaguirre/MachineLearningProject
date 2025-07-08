import streamlit as st
import pandas as pd
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os




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
import os

def cargar_modelo():
    return load_model("../Models/modelo_meteorologico.h5")

modelo = cargar_modelo()
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


import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Selector de fecha
fecha = st.date_input("Selecciona una fecha", value=None)

if fecha:
    fecha_str = fecha.strftime("%Y-%m-%d")


    bbox = "46.8,6.8,48.0,8.2"
    print(ciudad_seleccionada)

    # Coordenadas aproximadas de Basel
    if ciudad_seleccionada.lower() == "basel":
        bbox = "46.8,6.8,48.0,8.2"
    elif ciudad_seleccionada.lower() == "budapest":   
        bbox = "46.8,18.3,48.3,20.3"
    elif ciudad_seleccionada.lower() == "malmo":
        bbox = "55.2,12.4,56.0,13.6"
    elif ciudad_seleccionada.lower() == "roma":
        bbox = "41.2,11.5,42.3,13.0"
    elif ciudad_seleccionada.lower() == "heathrow":
        bbox = "51.2,-0.8,51.8,0.6"
    else:
        print("fallo")    
   


    # Construir URL de la imagen satelital
    url = f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&STYLES=&FORMAT=image/jpeg&CRS=EPSG:4326&BBOX={bbox}&WIDTH=512&HEIGHT=512&TIME={fecha_str}"

    # Descargar y mostrar
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        st.image(img, caption=f"Imagen satelital de Basel ({fecha_str})")
    else:
        st.error("No se pudo obtener la imagen para esa fecha.")



import requests
from PIL import Image
from io import BytesIO

#A partir de aquí es la predicción de si esta nublado o no, para ello se guarda la imagen satelital y la evalua con el modelo

if st.button("Predecir satelite"):

    def obtener_imagen_satelital(bbox, fecha):
        url = f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor",
            "STYLES": "",
            "FORMAT": "image/jpeg",
            "CRS": "EPSG:4326",
            "BBOX": bbox,
            "WIDTH": 256,
            "HEIGHT": 256,
            "TIME": fecha
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            st.error("No se pudo obtener la imagen satelital.")
            return None
        
    def predecir_con_modelo(imagen, modelo):
        img_resized = imagen.resize((64, 64)).convert("RGB")
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = modelo.predict(img_array)
        clases = ['despejado', 'muy_nublado', 'nublado', 'otros']
        clase = clases[np.argmax(pred)]#estoy sacando la clase de la prediccion
        confianza = np.max(pred)
        return clase, confianza

    imagen = obtener_imagen_satelital(bbox, fecha_str)

    if imagen:
        clase, confianza = predecir_con_modelo(imagen, modelo)
        st.markdown(f"### Predicción: **{clase}** ({confianza:.2%} de confianza)")
