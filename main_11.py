import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import requests
import pandas as pd
import numpy as np
import altair as alt

# ==============================
# CONFIG
# ==============================
load_dotenv(override=True)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API")

st.set_page_config(page_title="Razones financieras", layout="wide")

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
    return data

def calculate_financial_growth(df, columns):
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    df_sorted = df.sort_values(by='fiscalDateEnding').reset_index(drop=True)

    def calculate_yoy_growth_custom(current, previous):
        if pd.isna(previous) or previous == 0:
            return None
        if previous < 0:
            return ((current - previous) / abs(previous)) * 100
        else:
            return ((current - previous) / previous) * 100

    for col in columns:
        df_sorted[f'previous_{col}'] = df_sorted[col].shift(1)
        df_sorted[f'{col}_YoY_growth%'] = df_sorted.apply(
            lambda row: calculate_yoy_growth_custom(row[col], row[f'previous_{col}']),
            axis=1
        )
        df_sorted = df_sorted.drop(columns=[f'previous_{col}'])

    return df_sorted[['fiscalDateEnding'] + [f'{col}_YoY_growth%' for col in columns]]

# ==============================
# INTERFAZ DE USUARIO
# ==============================
st.title("📊 Razones financieras")
ticker = st.text_input("Ingrese el ticker de la empresa (ej. AAPL, MSFT):", value="AAPL").upper()

if ticker:
    st.write(f"Cargando datos para: **{ticker}** ...")
    
    @st.cache_data
    def load_financial_data(ticker):
        # Estado de resultados
        url_income = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}'
        income_data = get_data(url_income)
        income_df = pd.DataFrame.from_dict(income_data.get('annualReports', []))
        income_df = convert_columns_to_numeric(income_df, ['fiscalDateEnding'])

        # Balance general
        url_balance = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}'
        balance_data = get_data(url_balance)
        balance_df = pd.DataFrame.from_dict(balance_data.get('annualReports', []))
        balance_df = convert_columns_to_numeric(balance_df, ['fiscalDateEnding'])

        # Renombrar columnas para diferenciarlas
        balance_df = balance_df.rename(columns=lambda c: c if c == 'fiscalDateEnding' else f'b_{c}')
        income_df = income_df.rename(columns=lambda c: c if c == 'fiscalDateEnding' else f'i_{c}')

        # Merge de ambos
        finance_df = pd.merge(balance_df, income_df, on='fiscalDateEnding', how='left')
        finance_df = finance_df.fillna(0)
        return finance_df

    finance_df = load_financial_data(ticker)

    # ==============================
    # CÁLCULO DE RAZONES FINANCIERAS
    # ==============================
    # A) Rotación de inventario
    finance_df['previous_inventory'] = finance_df['b_inventory'].shift(-1)
    finance_df['average_inventory'] = (finance_df['b_inventory'] + finance_df['previous_inventory']) / 2
    finance_df = finance_df.drop(columns=['previous_inventory'])
    finance_df['rotacion_inventario'] = finance_df['i_costOfRevenue'] / finance_df['average_inventory']
    df_growth = calculate_financial_growth(finance_df.copy(), ['rotacion_inventario'])
    finance_df = pd.merge(finance_df, df_growth, on='fiscalDateEnding', how='left')

    # B) Rotación de cartera
    finance_df['previous_currentNetReceivables'] = finance_df['b_currentNetReceivables'].shift(-1)
    finance_df['average_currentNetReceivables'] = (finance_df['b_currentNetReceivables'] + finance_df['previous_currentNetReceivables']) / 2
    finance_df = finance_df.drop(columns=['previous_currentNetReceivables'])
    finance_df['rotacion_cartera'] = finance_df['i_totalRevenue'] / finance_df['average_currentNetReceivables']
    df_growth = calculate_financial_growth(finance_df.copy(), ['rotacion_cartera'])
    finance_df = pd.merge(finance_df, df_growth, on='fiscalDateEnding', how='left')

    # C) Razón circulante
    finance_df['razon_circulante'] = finance_df['b_totalCurrentAssets'] / finance_df['b_totalCurrentLiabilities']
    df_growth = calculate_financial_growth(finance_df.copy(), ['razon_circulante'])
    finance_df = pd.merge(finance_df, df_growth, on='fiscalDateEnding', how='left')

    # D) Prueba ácida
    finance_df['prueba_acida'] = (finance_df['b_totalCurrentAssets'] - finance_df['b_inventory']) / finance_df['b_totalCurrentLiabilities']
    df_growth = calculate_financial_growth(finance_df.copy(), ['prueba_acida'])
    finance_df = pd.merge(finance_df, df_growth, on='fiscalDateEnding', how='left')

    # E) Razón endeudamiento
    finance_df['razon_endeudamiento'] = finance_df['b_totalLiabilities'] / finance_df['b_totalShareholderEquity']
    df_growth = calculate_financial_growth(finance_df.copy(), ['razon_endeudamiento'])
    finance_df = pd.merge(finance_df, df_growth, on='fiscalDateEnding', how='left')

    # F) Razón solvencia
    finance_df['razon_solvencia'] = finance_df['b_totalLiabilities'] / finance_df['b_totalAssets']
    df_growth = calculate_financial_growth(finance_df.copy(), ['razon_solvencia'])
    finance_df = pd.merge(finance_df, df_growth, on='fiscalDateEnding', how='left')

    # ==============================
    # SELECCIÓN DE COLUMNAS
    # ==============================
    razones_financieras_df = finance_df[[
        'fiscalDateEnding',
        'rotacion_inventario', 'rotacion_inventario_YoY_growth%',
        'rotacion_cartera', 'rotacion_cartera_YoY_growth%',
        'razon_circulante', 'razon_circulante_YoY_growth%',
        'prueba_acida', 'prueba_acida_YoY_growth%',
        'razon_endeudamiento', 'razon_endeudamiento_YoY_growth%',
        'razon_solvencia', 'razon_solvencia_YoY_growth%'
    ]]
    
    st.subheader(f"📄 Razones financieras para {ticker}")
    st.dataframe(razones_financieras_df)

    # ==============================
    # GRÁFICO DE RAZONES FINANCIERAS
    # ==============================
    st.subheader("📈 Evolución de las razones financieras")
    razones_plot_df = razones_financieras_df.melt(
        id_vars='fiscalDateEnding', 
        value_vars=['rotacion_inventario', 'rotacion_cartera', 'razon_circulante', 'prueba_acida', 'razon_endeudamiento', 'razon_solvencia'],
        var_name='razon', 
        value_name='valor'
    )

    chart = alt.Chart(razones_plot_df).mark_line(point=True).encode(
        x='fiscalDateEnding:T',
        y='valor:Q',
        color='razon:N',
        tooltip=['fiscalDateEnding:T', 'razon:N', 'valor:Q']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)