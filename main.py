import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import requests
from typing import List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
load_dotenv(override=True)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API")

st.set_page_config(page_title="Panorama Macro", layout="wide")

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def convert_columns_to_numeric(df, columns_to_exclude=[]):
    for col in df.columns:
        if col not in columns_to_exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def get_data(url):
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame.from_dict(data["data"])

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

    # Bonos 2 años
    url_bono2 = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=2year&apikey={ALPHAVANTAGE_API_KEY}'
    bono2_df = get_data(url_bono2)
    bono2_df = convert_columns_to_numeric(bono2_df, ['date'])

    # Bonos 10 años
    url_bono10 = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={ALPHAVANTAGE_API_KEY}'
    bono10_df = get_data(url_bono10)
    bono10_df = convert_columns_to_numeric(bono10_df, ['date'])

    # WTI
    url_wti = f'https://www.alphavantage.co/query?function=WTI&interval=daily&apikey={ALPHAVANTAGE_API_KEY}'
    wti_df = get_data(url_wti)
    wti_df = convert_columns_to_numeric(wti_df, ['date'])

    # Brent
    url_brent = f'https://www.alphavantage.co/query?function=BRENT&interval=daily&apikey={ALPHAVANTAGE_API_KEY}'
    brent_df = get_data(url_brent)
    brent_df = convert_columns_to_numeric(brent_df, ['date'])

    # Oro
    url_gold = f'https://www.alphavantage.co/query?function=GOLD_SILVER_HISTORY&symbol=GOLD&interval=daily&apikey={ALPHAVANTAGE_API_KEY}'
    gold_df = get_data(url_gold)
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

    # Ajustes
    df = df.rename(columns={
        'wti_value': 'crudeoil_wti_value',
        'brent_value': 'crudeoil_brent_value',
        'gold_price': 'gold_price'
    })

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df[df["date"] >= "2012-01-01"]

    # Limpieza
    df[['bono2_value','bono10_value']] = df[['bono2_value','bono10_value']].ffill()

    cols_precios = ['crudeoil_wti_value','crudeoil_brent_value','gold_price']
    df[cols_precios] = df[cols_precios].ffill().interpolate()

    df.loc[df["crudeoil_wti_value"] < 0, "crudeoil_wti_value"] = np.nan
    df["crudeoil_wti_value"] = df["crudeoil_wti_value"].ffill()

    # ==============================
    # FEATURES
    # ==============================
    activos = {
        'wti': 'crudeoil_wti_value',
        'brent': 'crudeoil_brent_value',
        'bono2': 'bono2_value',
        'bono10': 'bono10_value',
        'gold': 'gold_price'
    }

    for n, col in activos.items():
        df[f'{n}_ret_1d'] = df[col].pct_change()
        df[f'{n}_ret_21d'] = df[col].pct_change(21)
        df[f'{n}_logret'] = np.log(df[col] / df[col].shift(1))
        df[f'{n}_ma20'] = df[col].rolling(20).mean()
        df[f'{n}_std20'] = df[col].rolling(20).std()
        df[f'{n}_zscore'] = (df[col] - df[f'{n}_ma20']) / df[f'{n}_std20']

    # spreads
    df['yield_curve'] = df['bono10_value'] - df['bono2_value']
    df['wti_brent_spread'] = df['crudeoil_wti_value'] - df['crudeoil_brent_value']

    # RSI
    for n, col in activos.items():
        df[f'{n}_rsi'] = rsi(df[col])

    return df.dropna()

# ==============================
# APP
# ==============================
import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ==============================
# CONFIG
# ==============================
load_dotenv(override=True)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API")

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

    # ==============================
    # FEATURES
    # ==============================
    activos = {
        'wti': 'crudeoil_wti_value',
        'brent': 'crudeoil_brent_value',
        'bono2': 'bono2_value',
        'bono10': 'bono10_value',
        'gold': 'gold_price'
    }

    for n, col in activos.items():
        df[f'{n}_ret_1d'] = df[col].pct_change()
        df[f'{n}_ret_21d'] = df[col].pct_change(21)
        df[f'{n}_logret'] = np.log(df[col] / df[col].shift(1))
        df[f'{n}_ma20'] = df[col].rolling(20).mean()
        df[f'{n}_std20'] = df[col].rolling(20).std()
        df[f'{n}_zscore'] = (df[col] - df[f'{n}_ma20']) / df[f'{n}_std20']

    df['yield_curve'] = df['bono10_value'] - df['bono2_value']
    df['wti_brent_spread'] = df['crudeoil_wti_value'] - df['crudeoil_brent_value']

    for n, col in activos.items():
        df[f'{n}_rsi'] = rsi(df[col])

    return df.dropna()

# ==============================
# APP
# ==============================
st.title("📊 Panorama Macro")

df = load_data()

if df.empty:
    st.stop()

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("WTI", round(df['crudeoil_wti_value'].iloc[-1],2))
col2.metric("Gold", round(df['gold_price'].iloc[-1],2))
col3.metric("Yield Curve", round(df['yield_curve'].iloc[-1],2))

# Gráfico
st.subheader("Precios")
st.line_chart(df.set_index("date")[[
    "crudeoil_wti_value",
    "crudeoil_brent_value",
    "gold_price"
]])

# Dataframe
st.subheader("Datos completos")
st.dataframe(df)