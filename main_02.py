# ========================================================================================================================
# CODIGO INDIVIDUAL: Noticias (prompt con open ai y finviz)
# ========================================================================================================================

# --------------------- Librerías ---------------------
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from typing import List
from pydantic import BaseModel, Field

# Librerías para noticias
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Descargar léxico si no existe
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --------------------- UI ---------------------
st.title("📊 Noticias Financieras con Finviz")

user_ticker = st.text_input("Ingresa el ticker (ej: AAPL, TSLA, JNJ)", value="JNJ")

# --------------------- OpenAI config ---------------------
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini"

# --------------------- Pydantic ---------------------
class GlobalInsights(BaseModel):
    preocupaciones: List[str] = Field(default_factory=list)
    avances: List[str] = Field(default_factory=list)

# --------------------- Función OpenAI ---------------------
def get_global_insights(titles: List[str]) -> GlobalInsights:
    combined_text = "\n".join(titles)

    prompt = f"""
Analiza las siguientes noticias financieras y económicas en español. 
Devuelve un JSON con dos campos:

1. 'preocupaciones': Lista de principales riesgos o problemas detectados.
2. 'avances': Lista de avances, fusiones o noticias importantes.

Noticias:
{combined_text}
"""

    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Eres un experto analista financiero. Devuelve SOLO JSON válido en español."
            },
            {"role": "user", "content": prompt},
        ],
        response_format=GlobalInsights
    )

    return response.choices[0].message.parsed

# --------------------- BOTÓN PRINCIPAL ---------------------
if st.button("Generar Resumen Global"):

    if not user_ticker:
        st.warning("Por favor ingresa un ticker válido.")
        st.stop()

    ticker = user_ticker.upper()

    # --------------------- Scraping ---------------------
    finviz_url = "https://finviz.com/quote.ashx?t="
    parsed_data = []

    try:
        url = finviz_url + ticker
        req = Request(url=url, headers={"user-agent": "app"})
        response = urlopen(req)

        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id="news-table")

        if news_table is None:
            st.error(f"No se encontraron noticias para {ticker}")
            st.stop()

        for row in news_table.find_all('tr'):
            if row.a is not None:
                title = row.a.text.strip()
                timestamp = row.td.text.split()

                if len(timestamp) == 1:
                    date = None
                    time_val = timestamp[0]
                else:
                    date = timestamp[0].lower()
                    time_val = timestamp[1]

                parsed_data.append([ticker, date, time_val, title])

    except Exception as e:
        st.error(f"Error al obtener datos: {e}")
        st.stop()

    # --------------------- DataFrame (interno, NO se muestra) ---------------------
    df_news = pd.DataFrame(parsed_data, columns=["ticker", "date", "time", "title"])

    # --------------------- Sentimiento ---------------------
    vader = SentimentIntensityAnalyzer()

    df_news['compound'] = df_news['title'].apply(lambda x: vader.polarity_scores(x)['compound'])

    df_news['sentiment'] = np.where(
        df_news['compound'] > 0, 'POS',
        np.where(df_news['compound'] < 0, 'NEG', 'NEU')
    )

    df_news.drop(columns=['compound'], inplace=True)

    # --------------------- Análisis con IA ---------------------
    st.subheader(f"📊 Análisis para {ticker}")

    with st.spinner("Analizando noticias..."):
        insights = get_global_insights(df_news["title"].tolist())

        st.subheader("⚠️ Principales Preocupaciones")
        if insights.preocupaciones:
            for p in insights.preocupaciones:
                st.write(f"- {p}")
        else:
            st.write("No se detectaron preocupaciones.")

        st.subheader("🚀 Avances o Progreso")
        if insights.avances:
            for a in insights.avances:
                st.write(f"- {a}")
        else:
            st.write("No se detectaron avances.")