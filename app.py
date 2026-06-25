import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Clasificador Gato vs Perro", layout="centered")

st.title("🐱 Clasificador de Gatos y Perros 🐶")
st.markdown("Sube una imagen para determinar si es un **gato** o un **perro**.")

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_gato_perro.h5")

modelo = cargar_modelo()
CLASES = ['Gato', 'Perro']

imagen_subida = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Clasificando..."):
        img_resized = imagen.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediccion = modelo.predict(img_array, verbose=0)
        clase = CLASES[np.argmax(prediccion)]
        confianza = np.max(prediccion) * 100

    if clase == "Gato":
        st.success(f"**Resultado: {clase}**")
    else:
        st.success(f"**Resultado: {clase}**")

    st.info(f"Confianza: {confianza:.2f}%")
