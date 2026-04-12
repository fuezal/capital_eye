# ==============================
# CODIGO INDIVIDUAL: DESCARGA DE CSV PANORAMA_DF
# ==============================

import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import requests
from typing import List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import time

# ==============================
# CONFIG
# ==============================
load_dotenv(override=True)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

st.set_page_config(page_title="Panorama Macro", layout="wide")


# ==============================
# FUNCIONES AUXILIARES
# ==============================
def convert_columns_to_numeric(df, columns_to_exclude=[]):
    for col in df.columns:
        if col not in columns_to_exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def get_data(url, name=""):
    try:
        response = requests.get(url)
        data = response.json()

        # 🔍 DEBUG opcional
        # st.write(data)

        # ❌ Manejo de errores API
        if "data" not in data:
            st.warning(f"⚠️ Error en {name}: {data}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data["data"])

        # ⏱️ Evitar rate limit
        time.sleep(12)

        return df

    except Exception as e:
        st.error(f"❌ Error cargando {name}: {e}")
        return pd.DataFrame()

def rsi(serie, ventana=14):
    delta = serie.diff()
    gain  = delta.clip(lower=0).rolling(ventana).mean()
    loss  = (-delta.clip(upper=0)).rolling(ventana).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ==============================
# DESCARGA DE DATOS
# ==============================
@st.cache_data
def load_data():

    # URLs
    url_bono2 = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=2year&apikey={ALPHAVANTAGE_API_KEY}'
    url_bono10 = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={ALPHAVANTAGE_API_KEY}'
    url_wti = f'https://www.alphavantage.co/query?function=WTI&interval=daily&apikey={ALPHAVANTAGE_API_KEY}'
    url_brent = f'https://www.alphavantage.co/query?function=BRENT&interval=daily&apikey={ALPHAVANTAGE_API_KEY}'
    url_gold = f'https://www.alphavantage.co/query?function=GOLD_SILVER_HISTORY&symbol=GOLD&interval=daily&apikey={ALPHAVANTAGE_API_KEY}'

    # Descarga (con protección)
    bono2_df = get_data(url_bono2, "bono 2Y")
    bono10_df = get_data(url_bono10, "bono 10Y")
    wti_df = get_data(url_wti, "WTI")
    brent_df = get_data(url_brent, "Brent")
    gold_df = get_data(url_gold, "Gold")

    # ⚠️ Validación mínima
    if bono2_df.empty or bono10_df.empty or wti_df.empty:
        st.error("❌ No se pudieron cargar datos críticos. Revisa API key o rate limit.")
        return pd.DataFrame()

    # Convertir
    bono2_df = convert_columns_to_numeric(bono2_df, ['date'])
    bono10_df = convert_columns_to_numeric(bono10_df, ['date'])
    wti_df = convert_columns_to_numeric(wti_df, ['date'])
    brent_df = convert_columns_to_numeric(brent_df, ['date'])
    gold_df = convert_columns_to_numeric(gold_df, ['date'])

    # Renombrar
    bono2_df = bono2_df.rename(columns=lambda c: c if c == 'date' else 'bono2_' + c)
    bono10_df = bono10_df.rename(columns=lambda c: c if c == 'date' else 'bono10_' + c)
    wti_df = wti_df.rename(columns=lambda c: c if c == 'date' else 'wti_' + c)
    brent_df = brent_df.rename(columns=lambda c: c if c == 'date' else 'brent_' + c)
    gold_df = gold_df.rename(columns=lambda c: c if c == 'date' else 'gold_' + c)

    # Merge
    df = pd.merge(wti_df, brent_df, on='date', how='left')
    df = pd.merge(df, bono2_df, on='date', how='left')
    df = pd.merge(df, bono10_df, on='date', how='left')
    df = pd.merge(df, gold_df, on='date', how='left')

    # Ajustes nombres
    df = df.rename(columns={
        'wti_value': 'crudeoil_wti_value',
        'brent_value': 'crudeoil_brent_value',
        'gold_price': 'gold_price'
    })

    # Fechas
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df[df["date"] >= "2012-01-01"]

    # Limpieza
    df[['bono2_value','bono10_value']] = df[['bono2_value','bono10_value']].ffill()

    cols_precios = ['crudeoil_wti_value','crudeoil_brent_value','gold_price']
    df[cols_precios] = df[cols_precios].ffill().interpolate()

    df.loc[df["crudeoil_wti_value"] < 0, "crudeoil_wti_value"] = np.nan
    df["crudeoil_wti_value"] = df["crudeoil_wti_value"].ffill()

    return df.dropna()

# ==============================
# APP
# ==============================
st.title("📊 Panorama Macro")

df = load_data()

# Dataframe
st.subheader("Datos completos")
st.dataframe(df)

descargar = st.checkbox("¿Quieres descargar los datos en CSV? 📥")

if descargar:
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name="panorama_df.csv",
        mime="text/csv"
    )