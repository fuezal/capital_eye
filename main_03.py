# --------------------- Librerías ---------------------
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import pandas as pd
from typing import List
from pydantic import BaseModel, Field

# --------------------- Cargar variables de entorno ---------------------
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------- Crear cliente de OpenAI ---------------------
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini"

# --------------------- Cargar dataset ---------------------
DATA_DIR = Path(__file__).resolve().parent
df_news = pd.read_csv(DATA_DIR / "df_news.csv")  # df_news debe tener columna "title"

# --------------------- Pydantic para el resumen global ---------------------
class GlobalInsights(BaseModel):
    preocupaciones: List[str] = Field(default_factory=list, description="Principales riesgos o problemas detectados")
    avances_salud: List[str] = Field(default_factory=list, description="Avances, fusiones o noticias del sector salud")

# --------------------- Función para resumen global ---------------------
def get_global_insights(titles: List[str]) -> GlobalInsights:
    combined_text = "\n".join(titles)
    prompt = f"""
Analiza las siguientes noticias financieras y económicas en español. 
Devuelve un JSON con dos campos:

1. 'preocupaciones': Lista de principales riesgos o problemas detectados.
2. 'avances_salud': Lista de avances, fusiones o noticias importantes relacionadas con el sector salud.

Noticias:
{combined_text}
"""
    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Eres un experto analista financiero y de mercado. Devuelve SOLO un JSON válido con los campos 'preocupaciones' y 'avances_salud'."
            },
            {"role": "user", "content": prompt},
        ],
        response_format=GlobalInsights
    )
    return response.choices[0].message.parsed

# --------------------- Streamlit UI ---------------------
st.title("📊 Resumen Global de Noticias Financieras")
st.write("Genera un análisis consolidado de todas las noticias, con riesgos y avances/fusiones en el sector salud.")

if st.button("Generar Resumen Global"):
    with st.spinner("Analizando todas las noticias..."):
        insights = get_global_insights(df_news["title"].tolist())
        
        st.subheader("⚠️ Principales Preocupaciones")
        if insights.preocupaciones:
            for p in insights.preocupaciones:
                st.write(f"- {p}")
        else:
            st.write("No se detectaron preocupaciones destacables.")
        
        st.subheader("🏥 Avances y Fusiones en Sector Salud")
        if insights.avances_salud:
            for a in insights.avances_salud:
                st.write(f"- {a}")
        else:
            st.write("No se detectaron avances o fusiones en el sector salud.")