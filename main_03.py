# SENTIMIENTO FINVIZ

# Librerías
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Descargar lexicón (solo la primera vez)
nltk.download('vader_lexicon')

# Configuración Streamlit
st.set_page_config(page_title="Sentimiento de Noticias", layout="wide")

# Título
st.title("📊 Análisis de Sentimiento de Noticias (Finviz)")

# URL base
FINVIZ_URL = "https://finviz.com/quote.ashx?t="

# Sidebar: selección de tickers
st.sidebar.header("Configuración")

default_tickers = ['JNJ','LLY','ABBV','MRK','ABT','ISRG','SYK','VRTX']
tickers_input = st.sidebar.text_input(
    "Tickers (separados por coma)",
    value=",".join(default_tickers)
)

tickers = [t.strip().upper() for t in tickers_input.split(",")]

# ==============================
# FUNCIÓN PARA OBTENER NOTICIAS
# ==============================
@st.cache_data(show_spinner=True)
def get_news(tickers):
    news_tables = {}

    for ticker in tickers:
        try:
            url = FINVIZ_URL + ticker
            req = Request(url=url, headers={"user-agent": "app"})
            response = urlopen(req)

            html = BeautifulSoup(response, "html.parser")
            news_table = html.find(id="news-table")
            news_tables[ticker] = news_table

        except Exception as e:
            st.warning(f"Error con {ticker}: {e}")

    parsed_data = []

    for ticker, news_table in news_tables.items():
        if news_table is None:
            continue

        for row in news_table.find_all('tr'):
            if row.a is not None:
                title = row.a.text
                timestamp = row.td.text.split()

                if len(timestamp) == 1:
                    date = None
                    time = timestamp[0]
                else:
                    date = timestamp[0]
                    time = timestamp[1]

                parsed_data.append([ticker, date, time, title])

    df = pd.DataFrame(parsed_data, columns=["ticker","date","time","title"])
    return df


# ==============================
# BOTÓN PARA EJECUTAR
# ==============================
if st.button("🔍 Analizar noticias"):

    with st.spinner("Obteniendo noticias..."):
        df_news = get_news(tickers)

    if df_news.empty:
        st.error("No se encontraron noticias.")
    else:
        # ==============================
        # ANÁLISIS DE SENTIMIENTO
        # ==============================
        vader = SentimentIntensityAnalyzer()

        df_news['compound'] = df_news['title'].apply(
            lambda x: vader.polarity_scores(x)['compound']
        )

        df_news['sentiment'] = np.where(
            df_news['compound'] > 0, 'POS',
            np.where(df_news['compound'] < 0, 'NEG', 'NEU')
        )

        df_news['title'] = df_news['title'].str.strip()
        df_news = df_news.drop(columns=['compound'])

        # ==============================
        # MOSTRAR DATOS
        # ==============================
        st.subheader("📰 Noticias")
        st.dataframe(df_news, use_container_width=True)

        # ==============================
        # GRÁFICA
        # ==============================
        st.subheader("📊 Distribución de Sentimiento")

        fig, ax = plt.subplots()
        sns.countplot(x='sentiment', data=df_news, ax=ax)
        ax.set_title("Distribución de Sentimiento")
        ax.set_xlabel("Sentimiento")
        ax.set_ylabel("Número de noticias")

        st.pyplot(fig)

        # ==============================
        # RESUMEN
        # ==============================
        st.subheader("📈 Resumen")

        resumen = df_news['sentiment'].value_counts().reset_index()
        resumen.columns = ['Sentimiento', 'Conteo']

        st.dataframe(resumen, use_container_width=True)

```
