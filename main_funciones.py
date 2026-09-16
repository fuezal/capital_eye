#Funciones para el análisis de datos
import streamlit as st
import pandas as pd



#Conversion de columnas a numericas
def convert_columns_to_numeric(df, columns_to_exclude=None):
    if columns_to_exclude is None:
        columns_to_exclude = []

    # Identify columns to convert to numeric
    columns_to_convert = df.columns.difference(columns_to_exclude)

    # Convert relevant columns to numeric, coercing errors
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


#calcula el crecimiento financiero
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


#imputa con la mediana
def impute_yoy_median(df):
    yoy_columns = [col for col in df.columns if 'YoY' in col]
    for col in yoy_columns:
        df[col] = df[col].fillna(df[col].median())
    return df

#convierte cifras a millones
def convert_columns_to_millions(df, columns_to_convert):
    # Convert specified columns to numeric, coercing errors and then fillna
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Divide the specified columns by 1,000,000
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = df[col] / 1_000_000

    return df

#redondea las cifras a 1 decimal
def round_numeric_df(df):
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].round(1)

    pd.set_option('display.float_format', '{:.1f}'.format)

    return df

# Redondea las cifras a 2 decimales 
def round_numeric_df2(df):
    numeric_cols = df.select_dtypes(include='number').columns

    df[numeric_cols] = df[numeric_cols].round(2)

    # Formato con coma para miles y 2 decimales
    pd.set_option('display.float_format', '{:,.2f}'.format)

    return df


# Redondea las cifras a 1 decimal y agrega separador de miles
# Si el decimal es 0, no lo muestra
def round_numeric_df1miles(df):
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        df[col] = df[col].round(1).map(
            lambda x: f"{x:,.0f}" if x.is_integer() else f"{x:,.1f}"
        )

    return df


#convierte cifras a miles de millones
def convert_columns_to_billions(df, columns_to_convert):
    # Convert specified columns to numeric, coercing errors and then fillna
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Divide the specified columns by 1,000,000,000
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = df[col] / 1000000000

    return df
