#Librerias obligatorias
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np

#graficos
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.figure_factory as ff

#cargar variables de entorno
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#crear cliente de openai
client = OpenAI(api_key=OPENAI_API_KEY)

#crear modelo de openai
model = "gpt-4o-mini"

_DATA_DIR = Path(__file__).resolve().parent
razones_financieras_df = pd.read_csv(_DATA_DIR / "razones_financieras_df.csv")

# Configuración inicial
st.set_page_config(layout="wide")

# ---------------------------
# FUNCIONES DE GRÁFICOS
# ---------------------------

# Gráfico 1: Ratios
def bar_plot1(df, column, title):
    fig = px.bar(
        df,
        x='fiscalDateEnding',
        y=column,
        title=title,
        text=df[column].apply(lambda x: f"{x:.2f}")
    )

    fig.update_traces(textposition='outside')

    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_title='Fecha',
        yaxis_title='Valor',
        showlegend=False
    )
    
    return fig


# Gráfico 2: Crecimiento (%)
def bar_plot2(df, column, title):
    fig = px.bar(
        df,
        x='fiscalDateEnding',
        y=column,
        title=title,
        text=df[column].apply(lambda x: f"{x:.2f}"),
        color=(df[column].fillna(0) > 0),
        color_discrete_map={
            True: '#00FF7F',
            False: '#FF3131'
        }
    )

    fig.update_traces(
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Valor: %{y:.2f}<extra></extra>'
    )

    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_title='Fecha',
        yaxis_title='Valor',
        showlegend=False
    )

    return fig


# ---------------------------
# UI
# ---------------------------

st.title("📊 Dashboard de Ratios Financieros J&J (Johnson & Johnson)")

# Mapeo: nombre bonito → nombre real
indicadores = {
    "Rotación de cartera": "rotacion_cartera",
    "Rotación de inventario": "rotacion_inventario",
    "Razón circulante": "razon_circulante",
    "Prueba ácida": "prueba_acida",
    "Razón de endeudamiento": "razon_endeudamiento",
    "Razón de solvencia": "razon_solvencia"
}

indicadores2 = {
    "Rotación de cartera (%)": "rotacion_cartera_YoY_growth%",
    "Rotación de inventario (%)": "rotacion_inventario_YoY_growth%",
    "Razón circulante (%)": "razon_circulante_YoY_growth%",
    "Prueba ácida (%)": "prueba_acida_YoY_growth%",
    "Endeudamiento (%)": "razon_endeudamiento_YoY_growth%",
    "Solvencia (%)": "razon_solvencia_YoY_growth%"
}

# Selectores
col_selector1, col_selector2 = st.columns(2)

with col_selector1:
    label1 = st.selectbox(
        "Indicador (Ratios)",
        list(indicadores.keys()),
        key="grafico1"
    )
    col = indicadores[label1]  # 👈 aquí conviertes

with col_selector2:
    label2 = st.selectbox(
        "Indicador (Crecimiento %)",
        list(indicadores2.keys()),
        key="grafico2"
    )
    col2 = indicadores2[label2]  # 👈 aquí conviertes

# Crear gráficos
fig1 = bar_plot1(razones_financieras_df, col, f"{label1}")
fig2 = bar_plot2(razones_financieras_df, col2, f"{label2}")

# Layout dashboard
col1, col2_layout = st.columns(2)

with col1:
    st.subheader("📈 Ratios Financieros")
    st.plotly_chart(fig1, use_container_width=True)

with col2_layout:
    st.subheader("📊 Crecimiento (%)")
    st.plotly_chart(fig2, use_container_width=True)