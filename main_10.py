import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

# ==============================
# CONFIG
# ==============================
load_dotenv(override=True)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") 

st.set_page_config(page_title="Razones financieras", layout="wide")

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def convert_columns_to_numeric(df, columns_to_exclude=[]):
    for col in df.columns:
        if col not in columns_to_exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_annual_reports(url, name=""):
    """Descarga annualReports de AlphaVantage (estructura distinta a /data)."""
    try:
        response = requests.get(url)
        data = response.json()

        if "annualReports" not in data:
            st.warning(f"⚠️ Error en {name}: {data}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data["annualReports"])
        time.sleep(12)  # evitar rate limit
        return df

    except Exception as e:
        st.error(f"❌ Error cargando {name}: {e}")
        return pd.DataFrame()


def calculate_financial_growth(df, columns):
    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
    df_sorted = df.sort_values(by="fiscalDateEnding").reset_index(drop=True)

    for col in columns:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors="coerce")

    def yoy(current, previous):
        if pd.isna(previous) or previous == 0:
            return None
        return ((current - previous) / abs(previous)) * 100

    for col in columns:
        df_sorted[f"previous_{col}"] = df_sorted[col].shift(1)
        df_sorted[f"{col}_YoY_growth%"] = df_sorted.apply(
            lambda row, c=col: yoy(row[c], row[f"previous_{c}"]), axis=1
        )
        df_sorted = df_sorted.drop(columns=[f"previous_{col}"])

    return df_sorted[["fiscalDateEnding"] + [f"{col}_YoY_growth%" for col in columns]]


# ==============================
# DESCARGA Y CÁLCULO
# ==============================
@st.cache_data
def load_data(ticker):
    api_key = ALPHAVANTAGE_API_KEY

    url_income = (
        f"https://www.alphavantage.co/query"
        f"?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}"
    )
    url_balance = (
        f"https://www.alphavantage.co/query"
        f"?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}"
    )

    year_income_df  = get_annual_reports(url_income,  "Income Statement")
    year_balance_df = get_annual_reports(url_balance, "Balance Sheet")

    if year_income_df.empty or year_balance_df.empty:
        return pd.DataFrame()

    year_income_df  = convert_columns_to_numeric(year_income_df,  ["fiscalDateEnding"])
    year_balance_df = convert_columns_to_numeric(year_balance_df, ["fiscalDateEnding"])

    year_balance_df = year_balance_df.rename(
        columns=lambda c: c if c == "fiscalDateEnding" else "b_" + c
    )
    year_income_df = year_income_df.rename(
        columns=lambda c: c if c == "fiscalDateEnding" else "i_" + c
    )

    year_balance_df["fiscalDateEnding"] = pd.to_datetime(year_balance_df["fiscalDateEnding"])
    year_income_df["fiscalDateEnding"]  = pd.to_datetime(year_income_df["fiscalDateEnding"])

    finance_df = pd.merge(year_balance_df, year_income_df, on="fiscalDateEnding", how="left")
    finance_df = finance_df.fillna(0)

    # A) Rotación de inventario
    finance_df["prev_inv"] = finance_df["b_inventory"].shift(-1)
    finance_df["average_inventory"] = (finance_df["b_inventory"] + finance_df["prev_inv"]) / 2
    finance_df = finance_df.drop(columns=["prev_inv"])
    finance_df["rotacion_inventario"] = finance_df["i_costOfRevenue"] / finance_df["average_inventory"]
    growth = calculate_financial_growth(finance_df.copy(), ["rotacion_inventario"])
    finance_df = pd.merge(finance_df, growth, on="fiscalDateEnding", how="left")

    # B) Rotación de cartera
    finance_df["prev_rec"] = finance_df["b_currentNetReceivables"].shift(-1)
    finance_df["average_receivables"] = (finance_df["b_currentNetReceivables"] + finance_df["prev_rec"]) / 2
    finance_df = finance_df.drop(columns=["prev_rec"])
    finance_df["rotacion_cartera"] = finance_df["i_totalRevenue"] / finance_df["average_receivables"]
    growth = calculate_financial_growth(finance_df.copy(), ["rotacion_cartera"])
    finance_df = pd.merge(finance_df, growth, on="fiscalDateEnding", how="left")

    # C) Razón circulante
    finance_df["razon_circulante"] = finance_df["b_totalCurrentAssets"] / finance_df["b_totalCurrentLiabilities"]
    growth = calculate_financial_growth(finance_df.copy(), ["razon_circulante"])
    finance_df = pd.merge(finance_df, growth, on="fiscalDateEnding", how="left")

    # D) Prueba ácida
    finance_df["prueba_acida"] = (
        finance_df["b_totalCurrentAssets"] - finance_df["b_inventory"]
    ) / finance_df["b_totalCurrentLiabilities"]
    growth = calculate_financial_growth(finance_df.copy(), ["prueba_acida"])
    finance_df = pd.merge(finance_df, growth, on="fiscalDateEnding", how="left")

    # E) Razón de endeudamiento
    finance_df["razon_endeudamiento"] = finance_df["b_totalLiabilities"] / finance_df["b_totalShareholderEquity"]
    growth = calculate_financial_growth(finance_df.copy(), ["razon_endeudamiento"])
    finance_df = pd.merge(finance_df, growth, on="fiscalDateEnding", how="left")

    # F) Razón de solvencia
    finance_df["razon_solvencia"] = finance_df["b_totalLiabilities"] / finance_df["b_totalAssets"]
    growth = calculate_financial_growth(finance_df.copy(), ["razon_solvencia"])
    finance_df = pd.merge(finance_df, growth, on="fiscalDateEnding", how="left")

    # Tabla final
    razones_df = finance_df[[
        "fiscalDateEnding",
        "rotacion_inventario", "rotacion_inventario_YoY_growth%",
        "rotacion_cartera",    "rotacion_cartera_YoY_growth%",
        "razon_circulante",    "razon_circulante_YoY_growth%",
        "prueba_acida",        "prueba_acida_YoY_growth%",
        "razon_endeudamiento", "razon_endeudamiento_YoY_growth%",
        "razon_solvencia",     "razon_solvencia_YoY_growth%",
    ]]
    razones_df = razones_df[razones_df["fiscalDateEnding"] >= "2008-12-31"]
    razones_df = razones_df.sort_values("fiscalDateEnding", ascending=False).reset_index(drop=True)
    return razones_df


# ==============================
# FUNCIONES DE GRÁFICOS
# ==============================
def bar_plot1(df, column, title):
    fig = px.bar(
        df, x="fiscalDateEnding", y=column, title=title,
        text=df[column].apply(lambda x: f"{x:.2f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white", title_x=0.5,
        xaxis_title="Fecha", yaxis_title="Valor", showlegend=False,
    )
    return fig


def bar_plot2(df, column, title):
    fig = px.bar(
        df, x="fiscalDateEnding", y=column, title=title,
        text=df[column].apply(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
        color=(df[column].fillna(0) > 0),
        color_discrete_map={True: "#00FF7F", False: "#FF3131"},
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Valor: %{y:.2f}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_white", title_x=0.5,
        xaxis_title="Fecha", yaxis_title="Valor (%)", showlegend=False,
    )
    return fig

# ==============================
# INTERFAZ: RAZONES FINANCIERAS
# ==============================
st.title("📊 Razones Financieras")

with st.sidebar:
    st.header("Parámetros")
    # Input del ticker
    ticker_input = st.text_input("Ticker", value="JNJ").upper()
    run = st.button("Cargar datos", type="primary")

# Guardar o reutilizar ticker en session_state
if "ticker" not in st.session_state:
    st.session_state["ticker"] = None
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()


# Actualizar ticker y cargar datos solo si se presiona el botón
if run:
    st.session_state["ticker"] = ticker_input
    with st.spinner(f"Descargando datos de {st.session_state['ticker']}..."):
        st.session_state["df"] = load_data(st.session_state["ticker"])
    
    if st.session_state["df"].empty:
        st.error("No se pudieron cargar datos. Revisa el ticker.")
        st.stop()

# Mostrar mensaje si no hay datos
if st.session_state["df"].empty:
    st.info("Ingresa un ticker en el panel izquierdo y presiona **Cargar datos**.")
else:
    df = st.session_state["df"]
    ticker = st.session_state["ticker"]
    st.success(f"**{ticker}** — {len(df)} años cargados")


    indicadores = {
        "Rotación de cartera":    "rotacion_cartera",
        "Rotación de inventario": "rotacion_inventario",
        "Razón circulante":       "razon_circulante",
        "Prueba ácida":           "prueba_acida",
        "Razón de endeudamiento": "razon_endeudamiento",
        "Razón de solvencia":     "razon_solvencia",
    }
    indicadores_growth = {
        "Rotación de cartera (%)":    "rotacion_cartera_YoY_growth%",
        "Rotación de inventario (%)": "rotacion_inventario_YoY_growth%",
        "Razón circulante (%)":       "razon_circulante_YoY_growth%",
        "Prueba ácida (%)":           "prueba_acida_YoY_growth%",
        "Endeudamiento (%)":          "razon_endeudamiento_YoY_growth%",
        "Solvencia (%)":              "razon_solvencia_YoY_growth%",
    }

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        label1 = st.selectbox("Indicador (Ratio)", list(indicadores.keys()), key="g1")
    with col_sel2:
        label2 = st.selectbox("Indicador (Crecimiento %)", list(indicadores_growth.keys()), key="g2")

    fig1 = bar_plot1(df, indicadores[label1], label1)
    fig2 = bar_plot2(df, indicadores_growth[label2], label2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Ratios Financieros")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("📊 Crecimiento YoY (%)")
        st.plotly_chart(fig2, use_container_width=True)