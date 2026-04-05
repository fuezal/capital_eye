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
    return pd.DataFrame.from_dict(data["data"])

def calculate_financial_growth(df, columns):
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    df_sorted = df.sort_values(by='fiscalDateEnding').reset_index(drop=True)

    # Convert specified columns to numeric, coercing errors
    for col in columns:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')

    # Define the custom Year-over-Year growth calculation function
    def calculate_yoy_growth_custom(current, previous):
        if pd.isna(previous) or previous == 0:
            return None
        if previous < 0:
            return ((current - previous) / abs(previous)) * 100
        else:
            return ((current - previous) / previous) * 100

    # Calculate Year-over-Year growth for each specified column using the custom function
    for col in columns:
        df_sorted[f'previous_{col}'] = df_sorted[col].shift(1)
        df_sorted[f'{col}_YoY_growth%'] = df_sorted.apply(
            lambda row: calculate_yoy_growth_custom(row[col], row[f'previous_{col}']),
            axis=1
        )
        df_sorted = df_sorted.drop(columns=[f'previous_{col}']) # Drop the temporary column

    return df_sorted[['fiscalDateEnding'] + [f'{col}_YoY_growth%' for col in columns]]

ticker ='JNJ'
# ==============================
# DESCARGA DE DATOS
# ==============================
@st.cache_data
def load_data():

    # Estado de resultado
    url_income = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={alphavantage_api}'
    response_income = get_data(url_income)
    income_sheet = response_income.json()
    year_income_df = pd.DataFrame.from_dict(income_sheet['annualReports'])
    year_income_df = convert_columns_to_numeric(year_income_df, ['fiscalDateEnding'])

    # s Balance general
    url_balance = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={alphavantage_api}'
    response_balance = get_data(url_balance)
    balance_sheet = response_balance.json()
    year_balance_df = pd.DataFrame.from_dict(balance_sheet['annualReports'])
    year_balance_df = convert_columns_to_numeric(year_balance_df, ['fiscalDateEnding'])


    # Renombrar
    balance_sheet_df = balance_sheet_df.rename(columns=lambda c: c if c == 'date' else 'b_' + c)
    year_income_df = year_income_df.rename(columns=lambda c: c if c == 'date' else 'i_' + c)

    #Convertir a formato fecha
    balance_sheet_df['fiscalDateEnding'] = pd.to_datetime(balance_sheet_df['fiscalDateEnding'])
    year_income_df['fiscalDateEnding'] = pd.to_datetime(year_income_df['fiscalDateEnding'])

    # Merge
    finance_df = pd.merge(balance_sheet_df, year_income_df, on='fiscalDateEnding', how='left')
   
   #valores nulos
   finance_df = finance_df.fillna(0)

   return df
  
#Variables creadas

# A) Rotacion de inventario (inventory_turnover)
#paso 1:
#Inventario promedio
finance_df['previous_inventory'] = finance_df['b_inventory'].shift(-1)
finance_df['average_inventory'] = (finance_df['b_inventory'] + finance_df['previous_inventory'])/2

# Eliminar columnas temporales
finance_df = finance_df.drop(columns=['previous_inventory'])

finance_df['rotacion_inventario'] = finance_df['i_costOfRevenue'] / finance_df['average_inventory']

#Crecimiento %
finance_df_with_growth = calculate_financial_growth(
    finance_df.copy(),
    ['rotacion_inventario']
)

#finance_df['fiscalDateEnding'] = pd.to_datetime(finance_df['fiscalDateEnding'])
finance_df_with_growth['fiscalDateEnding'] = pd.to_datetime(finance_df_with_growth['fiscalDateEnding'])
finance_df = pd.merge(finance_df, finance_df_with_growth, on='fiscalDateEnding', how='left')

#B) Rotacion de cartera (receivables_turnover)

# Promedio cuentas por cobrar
finance_df['previous_currentNetReceivables'] = finance_df['b_currentNetReceivables'].shift(-1)
finance_df['average_currentNetReceivables'] = (finance_df['b_currentNetReceivables'] + finance_df['previous_currentNetReceivables'])/2

# Eliminar columnas temporales
finance_df = finance_df.drop(columns=['previous_currentNetReceivables'])

# Rotacion de cartera (receivables_turnover)
finance_df['rotacion_cartera'] = finance_df['i_totalRevenue'] / finance_df['average_currentNetReceivables']

#Crecimiento %
finance_df_with_growth = calculate_financial_growth(
    finance_df.copy(),
    ['rotacion_cartera']
)

# Union de dataframes
finance_df = pd.merge(finance_df, finance_df_with_growth, on='fiscalDateEnding', how='left')

# C) Razon circulante

finance_df['razon_circulante'] = finance_df['b_totalCurrentAssets'] / finance_df['b_totalCurrentLiabilities']
#Crecimiento %
finance_df_with_growth = calculate_financial_growth(
    finance_df.copy(),
    ['razon_circulante']
)
# Union de dataframes
finance_df = pd.merge(finance_df, finance_df_with_growth, on='fiscalDateEnding', how='left')

# D) Prueba acida 

finance_df['prueba_acida'] = (finance_df['b_totalCurrentAssets']-finance_df['b_inventory'] )/ finance_df['b_totalCurrentLiabilities']
#Crecimiento %
finance_df_with_growth = calculate_financial_growth(
    finance_df.copy(),
    ['prueba_acida']
)
# Union de dataframes
finance_df = pd.merge(finance_df, finance_df_with_growth, on='fiscalDateEnding', how='left')

# E) Razon endeudamiento
finance_df['razon_endeudamiento'] = finance_df['b_totalLiabilities'] / finance_df['b_totalShareholderEquity']
#Crecimiento %
finance_df_with_growth = calculate_financial_growth(
    finance_df.copy(),
    ['razon_endeudamiento']
)
# Union de dataframes
finance_df = pd.merge(finance_df, finance_df_with_growth, on='fiscalDateEnding', how='left')

# F) Razon solvencia
finance_df['razon_solvencia'] = finance_df['b_totalLiabilities'] / finance_df['b_totalAssets']

#Crecimiento %
finance_df_with_growth = calculate_financial_growth(
    finance_df.copy(),
    ['razon_solvencia']
)
# Union de dataframes
finance_df = pd.merge(finance_df, finance_df_with_growth, on='fiscalDateEnding', how='left')

razones_financieras_df = finance_df[['fiscalDateEnding',
                                     'rotacion_inventario', 'rotacion_inventario_YoY_growth%',
                                     'rotacion_cartera', 'rotacion_cartera_YoY_growth%',
                                     'razon_circulante', 'razon_circulante_YoY_growth%',
                                     'prueba_acida', 'prueba_acida_YoY_growth%',
                                     'razon_endeudamiento', 'razon_endeudamiento_YoY_growth%',
                                     'razon_solvencia', 'razon_solvencia_YoY_growth%']]

razones_financieras_df = razones_financieras_df[razones_financieras_df["fiscalDateEnding"] >= "2008-12-31"]