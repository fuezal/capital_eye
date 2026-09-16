###############################################################################################
####### Calculadora de analisis fundamental
###############################################################################################

# Como la API de alpha vantage es de uso gratuito, solo se permiten 25 consultas a los datos,
# cada que se analiza una accion son 4 consultas a la vez debido a los estados financieros que
# se mandan a llamar: Flujo de Efectivo, Balance, Estado de Resultados, Earnings

#librerias_______________________________________________________________________________
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px #graficos
import plotly.graph_objects as go
import textwrap
import streamlit.components.v1 as components
from datetime import datetime
import os
import time
import requests
from dotenv import load_dotenv

#Funciones_______________________________________________________________________________
from main_funciones import convert_columns_to_numeric
from main_funciones import calculate_financial_growth
from main_funciones import impute_yoy_median
from main_funciones import convert_columns_to_millions
from main_funciones import round_numeric_df
from main_funciones import round_numeric_df1miles
from main_funciones import convert_columns_to_billions

from prompts import render_noticias_ui

#API______________________________________________________________________________________
load_dotenv(override=True)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

#Nombre de la pagina _____________________________________________________________________
st.set_page_config(page_title="gabriela fuentes", layout="wide", initial_sidebar_state="expanded")


# CALCULADORA_____________________________________________________________________________

# PALETA DE COLORES
PURPLE_DARK  = "#2A1E8C"
PURPLE_MID   = "#4B3AD6"
PURPLE_DEEP  = "#7C6CF5"
ITEM_HOVER   = "#6F5FF0"
ITEM_ACTIVE  = "#7C6CF5"
WHITE        = "#FFFFFF"
WHITE_MUTED  = "rgba(255,255,255,0.75)"
CARD_BG      = "#F0F0F3"

# MENÚ

MENU_ITEMS = [
    ("dashboard", "Dashboard"),
    ("Finanzas clave",  "Finanzas clave"),
    ("graficos",  "Más gráficos"),
    ("noticias", "Noticias"),
    ("glosario", "Glosario"),
]

if "menu_selection" not in st.session_state:
    st.session_state.menu_selection = "dashboard"

current = st.session_state.menu_selection
#active_index = [item_id for item_id, _ in MENU_ITEMS].index(current) + 1  # 1-based para nth-of-type

# ------------------------------------------------------------------
# DISEÑO
# ------------------------------------------------------------------

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
/* Sidebar: ancho fijo, morado, alto completo (el propio section ya
   se extiende a toda la altura del viewport por defecto en Streamlit) */
section[data-testid="stSidebar"] {{
    background: linear-gradient(160deg, {PURPLE_DARK} 0%, {PURPLE_MID} 55%, {PURPLE_DEEP} 100%); #color del menu
    min-width: 200px !important;
    max-width: 200px !important;
}}


section[data-testid="stSidebar"] > div {{
    padding-top: 1.5rem;
    background: transparent;
}}

/* Título "Buscar Ticker" */
.ticker-label {{
    color: {WHITE};
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin: 0 0 8px 0;
    padding: 0 1.2rem;
}}

/* Input de texto (ticker) - widget nativo de Streamlit */
section[data-testid="stSidebar"] div[data-testid="stTextInput"] {{
    padding: 0 1.2rem;
    margin-bottom: 24px;
}}
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {{
    background-color: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    border-radius: 10px !important;
    color: #000000 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
}}
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input::placeholder {{
    color: {WHITE_MUTED} !important;
}}

/* Separador entre buscador y menú */
.sidebar-divider {{
    border: none;
    border-top: 1px solid rgba(255,255,255,0.18);
    margin: 0 1.2rem 12px 1.2rem;
}}

/* -----------------------------------------------------------
   Botones del menú (st.button) estilizados como items de lista
   ----------------------------------------------------------- */
section[data-testid="stSidebar"] div[data-testid="stButton"] {{
    padding: 0 0.8rem;
    margin-bottom: 4px;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    border: none;
    box-shadow: none;
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 15px;
    transition: background-color 0.15s ease;
}}

/* Botón NO seleccionado (secondary) */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] {{
    background-color: transparent !important;
    color: {WHITE} !important;
    font-weight: 500;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover {{
    background-color: {ITEM_HOVER} !important;
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:focus {{
    box-shadow: none !important;
    color: {WHITE} !important;
}}

/* Botón de la sección ACTIVA (primary) -> sombreado */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {{
    background-color: {ITEM_ACTIVE} !important;
    color: {WHITE} !important;
    font-weight: 700;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {{
    background-color: {ITEM_ACTIVE} !important;
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:focus {{
    box-shadow: none !important;
    color: {WHITE} !important;
}}

/* Fondo general del área de contenido */
[data-testid="stAppViewContainer"] > .main {{
    background-color: #F3F3FA;
}}
/* Reducir el espacio superior del contenido principal */
.block-container {{
    padding-top: 2rem !important;
}}

/* -----------------------------------------------------------
   Tarjetas (cards) de resumen
   ----------------------------------------------------------- */
.cards-row {{
    display: flex;
    flex-direction: row;
    gap: 20px;
    margin-bottom: 18px;
}}

.info-card {{
    flex: 1;
    background-color: {CARD_BG};
    border-radius: 16px;
    padding: 16px 20px;
    min-height: 6px;
}}

.card-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}}

.card-icon {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background-color: rgba(124,108,245,0.12);
    color: {ITEM_ACTIVE};
    font-size: 16px;
    flex-shrink: 0;
}}


.info-card .card-label {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #9A9AA5;
    margin-bottom: 8px;
}}
.info-card .card-value {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #1F1F1F;
    word-break: break-word;
}}

.info-card .card-label {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #9A9AA5;
    margin-bottom: 8px;
}}
.info-card .card-value {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #1F1F1F;
    word-break: break-word;
}}

/* ================= NUEVO: tarjeta Sector (1/4 ancho, izquierda) ================= */
.wide-card-row {{
    display: flex;
    flex-direction: row;
    margin-bottom: 18px;
}}
.info-card-wide {{
    background-color: {CARD_BG};
    border-radius: 16px;
    padding: 18px 24px;
    width: 100%;
    box-sizing: border-box;


}}
.info-card-wide .card-label {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #9A9AA5;
    margin-bottom: 4px;
}}
.info-card-wide .card-sector {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 18px;
    font-weight: 800;
    color: #1F1F1F;
    margin-bottom: 12px;
}}
.info-card-wide .card-desc-list {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 13px;
    color: #4A4A4A;
    margin: 0;
    padding-left: 18px;
    line-height: 1.5;
}}
.info-card-wide .card-desc-list li {{
    break-inside: avoid;
}}
/* ================= FIN NUEVO ================= */

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Contenedor que fusiona el selectbox con la tarjeta de abajo */
.metric-picker-wrap {
    background: #F7F7F9;
    border: 1px solid #eee;
    border-bottom: none;
    border-radius: 18px 18px 0 0;
    padding: 14px 18px 0 18px;
    margin-bottom: -14px;
}
.metric-picker-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #9A9AA5;
    margin-bottom: 6px;
}

/* Restyle del selectbox nativo de Streamlit dentro de este contenedor */
div[data-testid="stSelectbox"] {
    max-width: 260px;
}
div[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #C9BFF7 !important;
    border-radius: 999px !important;
    padding: 2px 6px !important;
    box-shadow: none !important;
}
div[data-testid="stSelectbox"] > div > div:hover {
    border-color: #7C6CF5 !important;
}
div[data-testid="stSelectbox"] label {
    display: none;
}
div[data-testid="stSelectbox"] svg {
    color: #7C6CF5 !important;
}
div[data-testid="stSelectbox"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #4B3AD6 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# SIDEBAR
#1.Lectura y consulta a la API de acuerdo al ticker solicitado
#2. Diseño de dataset
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="ticker-label">Buscar Ticker</div>', unsafe_allow_html=True)
    ticker = st.text_input(
        "Buscar Ticker",
        value="MSFT",
        placeholder="Ej. MSFT",
        label_visibility="collapsed",
        key="ticker_input",
    ).upper()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)


    for item_id, label in MENU_ITEMS:
        btn_type = "primary" if item_id == current else "secondary"
        if st.button(label, key=f"menu_{item_id}", use_container_width=True, type=btn_type):
            st.session_state.menu_selection = item_id
            st.rerun()

# Procedimiento con el dataset listo para analizar
if not ALPHAVANTAGE_API_KEY:
    st.error(
        "No se encontró ALPHAVANTAGE_API_KEY. Defínela en tu archivo .env "
        "o en los 'Secrets' de Streamlit."
    )
    st.stop()


if not ticker:
    st.info("Escribe un ticker para continuar.")
    st.stop()


def _es_nota_de_limite_de_tasa(note: str) -> bool:
    """Detecta cualquier variante del aviso de límite de peticiones de AlphaVantage."""
    if not note:
        return False
    frases_limite = [
        "burst pattern detected",
        "spreading out your free api requests",
        "per minute",
        "per second",
        "standard api rate limit",
        "premium plans",
    ]
    note_lower = note.lower()
    return any(frase in note_lower for frase in frases_limite)


def _fetch_alpha_vantage_data(url, description, ticker, max_retries=3, backoff_seconds=15):
    """Obtiene datos de AlphaVantage con reintentos que respetan el límite de peticiones."""
    payload = None
    for attempt in range(max_retries):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        note = payload.get("Note") or payload.get("Information")

        if note and _es_nota_de_limite_de_tasa(note):
            # Si el mensaje indica el límite DIARIO (25/día), reintentar no sirve de nada.
            if "25 requests per day" in note and "day" in note.lower() and attempt == 0:
                # Igual probamos una vez más por si fue solo el límite por segundo,
                # pero avisamos con un mensaje claro si vuelve a fallar.
                pass
            if attempt == max_retries - 1:
                raise ValueError(
                    f"{description} ({ticker}) alcanzó el límite de peticiones de AlphaVantage.\n\n"
                    f"Mensaje original: {note}\n\n"
                    "Si ya hiciste 25 peticiones hoy con tu API key gratuita, tendrás que "
                    "esperar hasta mañana o usar un plan premium. Si no, espera unos "
                    "segundos e inténtalo de nuevo."
                )
            time.sleep(backoff_seconds * (attempt + 1))
            continue
        if note:
            # Otro tipo de mensaje informativo/error (p. ej. ticker inválido)
            raise ValueError(f"{description} ({ticker}): {note}")
        return payload
    return payload


@st.cache_data(ttl=86400, show_spinner="Descargando datos financieros...")
def cargar_datos_financieros(ticker: str) -> pd.DataFrame:
    base_url = "https://www.alphavantage.co/query"

    url_income = f"{base_url}?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    url_balance = f"{base_url}?function=BALANCE_SHEET&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    url_cashflow = f"{base_url}?function=CASH_FLOW&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    url_earnings = f"{base_url}?function=EARNINGS&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"

    # Espaciamos las llamadas 1 segundo para respetar el límite por segundo de AlphaVantage.
    income_statement = _fetch_alpha_vantage_data(url_income, "Estado de resultados", ticker)
    time.sleep(1)
    balance_sheet = _fetch_alpha_vantage_data(url_balance, "Balance general", ticker)
    time.sleep(1)
    cashflow_statement = _fetch_alpha_vantage_data(url_cashflow, "Flujo de caja", ticker)
    time.sleep(1)
    earnings_history = _fetch_alpha_vantage_data(url_earnings, "Historial de ganancias", ticker)

    # Validación de estructura esperada
    for nombre, payload, clave in [
        ("Estado de resultados", income_statement, "annualReports"),
        ("Balance general", balance_sheet, "annualReports"),
        ("Flujo de caja", cashflow_statement, "annualReports"),
        ("Historial de ganancias", earnings_history, "annualEarnings"),
    ]:
        if clave not in payload:
            raise ValueError(
                f"{nombre} no devolvió '{clave}' para el ticker '{ticker}'. "
                f"Respuesta recibida: {payload}"
            )

    income_df_trim = pd.DataFrame.from_dict(income_statement["quarterlyReports"])  # trimestral
    income_df = pd.DataFrame.from_dict(income_statement["annualReports"])  # anual

    balance_df = pd.DataFrame.from_dict(balance_sheet["annualReports"])
    cashflow_df = pd.DataFrame.from_dict(cashflow_statement["annualReports"])
    earnings_df = pd.DataFrame.from_dict(earnings_history["annualEarnings"])

    # Formato numérico
    income_df_trim = convert_columns_to_numeric(
        income_df_trim, columns_to_exclude=["fiscalDateEnding", "reportedCurrency"]
    )
    income_df = convert_columns_to_numeric(
        income_df, columns_to_exclude=["fiscalDateEnding", "reportedCurrency"]
    )
    balance_df = convert_columns_to_numeric(
        balance_df, columns_to_exclude=["fiscalDateEnding", "reportedCurrency"]
    )
    cashflow_df = convert_columns_to_numeric(
        cashflow_df, columns_to_exclude=["fiscalDateEnding", "reportedCurrency"]
    )
    earnings_df["reportedEPS"] = pd.to_numeric(earnings_df["reportedEPS"], errors="coerce")

    # Formato de fecha
    income_df_trim["fiscalDateEnding"] = pd.to_datetime(income_df_trim["fiscalDateEnding"])
    income_df["fiscalDateEnding"] = pd.to_datetime(income_df["fiscalDateEnding"])
    balance_df["fiscalDateEnding"] = pd.to_datetime(balance_df["fiscalDateEnding"])
    cashflow_df["fiscalDateEnding"] = pd.to_datetime(cashflow_df["fiscalDateEnding"])
    earnings_df["fiscalDateEnding"] = pd.to_datetime(earnings_df["fiscalDateEnding"])

    # Merge estado de resultados + ganancias
    income_df = pd.merge(income_df, earnings_df, on="fiscalDateEnding", how="left")

    income_df.columns = [
        col if col == "fiscalDateEnding" else "i_" + col for col in income_df.columns
    ]
    balance_df.columns = [
        col if col == "fiscalDateEnding" else "b_" + col for col in balance_df.columns
    ]
    cashflow_df.columns = [
        col if col == "fiscalDateEnding" else "c_" + col for col in cashflow_df.columns
    ]

    finance_df = pd.merge(income_df, balance_df, on="fiscalDateEnding", how="left")
    finance_df = pd.merge(finance_df, cashflow_df, on="fiscalDateEnding", how="left")

    # Guardamos también el trimestral por si se usa en un diagrama
    st.session_state["income_df_trim"] = income_df_trim

    return finance_df


try:
    finance_df = cargar_datos_financieros(ticker)
except ValueError as e:
    st.error(str(e))
    st.stop()
except requests.exceptions.RequestException as e:
    st.error(f"Error de conexión con AlphaVantage: {e}")
    st.stop()


finance_df['fiscalDateEnding'] = pd.to_datetime(finance_df['fiscalDateEnding'])




# Ingenieria de variables _________________________________________________________________

# Eliminar columnas que no son relevantes para el análisis
columns_to_drop = [
    'i_reportedCurrency',
    'i_investmentIncomeNet',
    'i_nonInterestIncome',
    'i_depreciation',
    'i_interestAndDebtExpense',
    'i_comprehensiveIncomeNetOfTax',
    'b_reportedCurrency',
    'b_accumulatedDepreciationAmortizationPPE',
    'b_investments',
    'b_otherNonCurrentAssets',
    'b_currentDebt',
    'b_longTermDebtNoncurrent',
    'b_treasuryStock',
    'c_reportedCurrency',
    'c_paymentsForOperatingActivities',
    'c_proceedsFromOperatingActivities',
    'c_changeInOperatingLiabilities',
    'c_changeInOperatingAssets',
    'c_profitLoss',
    'c_proceedsFromRepaymentsOfShortTermDebt',
    'c_paymentsForRepurchaseOfEquity',
    'c_proceedsFromIssuanceOfCommonStock',
    'c_proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet',
    'c_proceedsFromIssuanceOfPreferredStock',
    'c_proceedsFromSaleOfTreasuryStock',
    'c_netIncome'
]

finance_df = finance_df.drop(columns=columns_to_drop)

# Imputar valores nulos con 0
finance_df['i_reportedEPS'] = finance_df['i_reportedEPS'].fillna(0)
finance_df['i_interestIncome'] = finance_df['i_interestIncome'].fillna(0)
c_columns = [col for col in finance_df.columns if col.startswith('c_')]
finance_df[c_columns] = finance_df[c_columns].fillna(0)

#Variables nuevas

# EBITDA Margin
finance_df['ebitdaMargin%'] = finance_df['i_ebitda']*100 / finance_df['i_totalRevenue']
# EBIT Margin
finance_df['ebitMargin%'] = finance_df['i_operatingIncome']*100 / finance_df['i_totalRevenue']
# Net income Margin
finance_df['netIncomeMargin%'] = finance_df['i_netIncome']*100 / finance_df['i_totalRevenue']
# Net Interest
finance_df['netInterest'] = finance_df['i_interestIncome'] - finance_df['i_interestExpense']
# Tax Rate
finance_df['taxRate%'] = finance_df['i_incomeTaxExpense']*100 / finance_df['i_incomeBeforeTax']
# Tax Rate is 21% in USA, se imputa en caso de que no haya datos
finance_df['taxRate%'] = finance_df['taxRate%'].fillna(21.0)

columns_to_grow = ['i_totalRevenue','i_grossProfit','i_operatingIncome','i_ebitda',
                   'i_netIncome','i_reportedEPS','b_commonStockSharesOutstanding']
grow_df = calculate_financial_growth(finance_df.copy(), columns_to_grow)


#grow_df['fiscalDateEnding'] = pd.to_datetime(grow_df['fiscalDateEnding'])

# Union de dataframes para tener el crecimiento financiero
finance_df = pd.merge(finance_df, grow_df, on='fiscalDateEnding', how='left')

# se imputa en caso de que no haya datos
columns_to_impute = ['i_reportedEPS_YoY_growth%']
finance_df[columns_to_impute] = finance_df[columns_to_impute].fillna(0)
finance_df = impute_yoy_median(finance_df)
finance_df = finance_df.fillna(0)

#Variables nuevas
finance_df['maintenanceCapex'] = finance_df.apply(
    lambda row: row['c_capitalExpenditures'] if row['c_capitalExpenditures'] < row['c_depreciationDepletionAndAmortization']
    else row['c_depreciationDepletionAndAmortization'],
    axis=1
)

finance_df['growthCapex'] = finance_df['c_capitalExpenditures'] - finance_df['maintenanceCapex']

# Reemplaza 'deferredRevenue' con 'otherNonCurrentLiabilities' si 'deferredRevenue' es negativo

finance_df['b_deferredRevenue'] = finance_df.apply(
    lambda row: row['b_otherNonCurrentLiabilities'] if row['b_deferredRevenue'] <= 0 else row['b_deferredRevenue'],
    axis=1
)

# eliminar columna
finance_df = finance_df.drop(columns=['b_otherNonCurrentLiabilities'])

# imputar con la mediana para 'deferredRevenue' en caso de que sea necesario
if finance_df['b_deferredRevenue'].isnull().any():
    median_deferredRevenue = finance_df['b_deferredRevenue'].median()
    finance_df['b_deferredRevenue'] = finance_df['b_deferredRevenue'].fillna(median_deferredRevenue)

finance_df['c_ebitda'] = finance_df['i_operatingIncome'] + finance_df['c_depreciationDepletionAndAmortization']

finance_df['workingCapital'] = finance_df['b_inventory'] + finance_df['b_currentNetReceivables'] - finance_df['b_currentAccountsPayable'] - finance_df['b_deferredRevenue']

#Columnas tempporales
finance_df['previous_currentAccountsPayable'] = finance_df['b_currentAccountsPayable'].shift(-1)
finance_df['previous_workingCapital'] = finance_df['workingCapital'].shift(-1)

finance_df['changeInWC'] = finance_df.apply(lambda row: (row['workingCapital'] - row['previous_workingCapital']) if row['previous_currentAccountsPayable'] > 0 else 0, axis=1)

# Drop the temporary columns
finance_df = finance_df.drop(columns=['previous_currentAccountsPayable', 'previous_workingCapital'])

#Variables nuevas
finance_df['freeCashFlow'] = finance_df['c_ebitda'] - finance_df['i_interestExpense'] - finance_df['i_incomeTaxExpense'] - finance_df['maintenanceCapex'] - finance_df['changeInWC']
finance_df['freeCashFlowPerShare'] = finance_df['freeCashFlow'] / finance_df['b_commonStockSharesOutstanding']
fcf_df_with_growth = calculate_financial_growth(finance_df.copy(), ['freeCashFlow','freeCashFlowPerShare'])

# Union de dataframes
finance_df = pd.merge(finance_df, fcf_df_with_growth, on='fiscalDateEnding', how='left')

# variables
finance_df['maintenanceCapexMargin%'] = finance_df['maintenanceCapex']*100 / finance_df['i_totalRevenue']
finance_df['workingCapitalMargin%'] = finance_df['workingCapital']*100 / finance_df['i_totalRevenue']
finance_df['freeCashFlowMargin%'] = finance_df['freeCashFlow']*100 / finance_df['i_totalRevenue']
finance_df['cashConversion%'] = finance_df['freeCashFlow']*100 / finance_df['c_ebitda']

# se imputa en caso de que no haya datos
finance_df = impute_yoy_median(finance_df)
finance_df = finance_df.fillna(0)

# Variables nuevas
finance_df['investedCapital'] = finance_df['b_totalShareholderEquity'] + finance_df['b_shortTermDebt'] + finance_df['b_longTermDebt'] + finance_df['b_capitalLeaseObligations'] - finance_df['b_shortTermInvestments']

#Metodo al cierre del año
#finance_df['ROA%'] = finance_df['i_netIncome']*100 / finance_df['b_totalAssets']
#finance_df['ROE%'] = finance_df['i_netIncome']*100 / finance_df['b_totalShareholderEquity']

#Metodo con promedio 
def safe_avg_ratio(numerator_col, denominator_col, df, shift_periods=-1, multiply_100=True):
    try:
        promedio = (df[denominator_col] + df[denominator_col].shift(shift_periods)) / 2
        resultado = df[numerator_col] * 100 / promedio if multiply_100 else df[numerator_col] / promedio
        resultado = resultado.replace([np.inf, -np.inf], np.nan)
        return resultado
    except Exception as e:
        print(f"[safe_avg_ratio] Error calculando {numerator_col}/{denominator_col}: {e}")
        return pd.Series([np.nan] * len(df), index=df.index)


# --- ROA% con promedio ---
finance_df['b_totalAssets_avg'] = (finance_df['b_totalAssets'] + finance_df['b_totalAssets'].shift(-1)) / 2
finance_df['ROA%'] = safe_avg_ratio('i_netIncome', 'b_totalAssets', finance_df)

# --- ROE% con promedio ---
finance_df['ROE%'] = safe_avg_ratio('i_netIncome', 'b_totalShareholderEquity', finance_df)

finance_df['NOPAT'] = finance_df['i_operatingIncome'] * (1 - finance_df['taxRate%'] / 100)
finance_df['ROIC%'] = finance_df['NOPAT']*100 / finance_df['investedCapital']

# Replace infinite values with 0 in the specified columns
for col in ['ROA%', 'ROE%', 'ROIC%']:
    finance_df[col] = finance_df[col].replace([np.inf, -np.inf], 0)

# Variables nuevas
finance_df['reinvestmentRate%'] = finance_df.apply(lambda row: row['growthCapex']*100/row['freeCashFlow'] if row['freeCashFlow'] > 0 else 0, axis=1)
finance_df = finance_df.fillna(0)

# KPIS -------------------------------------------------------------
finance_df['dividendPayout'] = finance_df['c_dividendPayoutCommonStock'] + finance_df['c_dividendPayoutPreferredStock']
finance_df['paymentsForRepurchase'] = finance_df['c_paymentsForRepurchaseOfCommonStock'] + finance_df['c_paymentsForRepurchaseOfPreferredStock']
finance_df['debt'] = finance_df['b_shortTermDebt'] + finance_df['b_longTermDebt']

finance_df['debtRepayment'] = finance_df['debt'].diff(-1) * -1
finance_df.loc[finance_df['debtRepayment'] < 0, 'debtRepayment'] = 0
finance_df['debtRepayment'] = finance_df['debtRepayment'].fillna(0)

finance_df['organicGrowth%'] = finance_df['growthCapex']*100 / finance_df['freeCashFlow']
finance_df['dividendPayout%'] = finance_df['dividendPayout']*100 / finance_df['freeCashFlow']
finance_df['buybacks%'] = finance_df['paymentsForRepurchase']*100 / finance_df['freeCashFlow']
finance_df['debtAmortization%'] = finance_df['debtRepayment']*100 / finance_df['freeCashFlow']
finance_df['total%'] = finance_df['organicGrowth%']+finance_df['dividendPayout%']+finance_df['buybacks%']+finance_df['debtAmortization%']

#Convertir valores negativos a 0
kpis_numeric_cols = [
    'organicGrowth%',
    'dividendPayout%',
    'buybacks%',
    'debtAmortization%',
    'total%'
]
finance_df[kpis_numeric_cols] = finance_df[kpis_numeric_cols].clip(lower=0)

# Columnas a eliminar
columns_to_drop = [
    'i_costofGoodsAndServicesSold',
    'i_sellingGeneralAndAdministrative',
    'i_researchAndDevelopment',
    'i_operatingExpenses',
    'i_netInterestIncome',
    'i_otherNonOperatingIncome',
    'i_depreciationAndAmortization',
    'i_netIncomeFromContinuingOperations',
    'i_ebit',
    'b_totalAssets',
    'b_cashAndCashEquivalentsAtCarryingValue',
    'b_totalNonCurrentAssets',
    'b_propertyPlantEquipment',
    'b_intangibleAssets',
    'b_intangibleAssetsExcludingGoodwill',
    'b_goodwill',
    'b_longTermInvestments',
    'b_otherCurrentAssets',
    'b_totalLiabilities',
    'b_totalNonCurrentLiabilities',
    'b_shortLongTermDebtTotal',
    'b_otherCurrentLiabilities',
    'b_retainedEarnings',
    'b_commonStock',
    'c_operatingCashflow',
    'c_changeInReceivables',
    'c_changeInInventory',
    'c_cashflowFromInvestment',
    'c_cashflowFromFinancing',
    'c_paymentsForRepurchaseOfCommonStock',
    'c_paymentsForRepurchaseOfPreferredStock',
    'c_dividendPayout',
    'c_dividendPayoutCommonStock',
    'c_dividendPayoutPreferredStock',
    'c_proceedsFromRepurchaseOfEquity',
    'c_stockBasedCompensation',
    'c_changeInCashAndCashEquivalents',
    'c_changeInExchangeRate',
    'i_grossProfit_YoY_growth%',
    'NOPAT',
    'dividendPayout',
    'paymentsForRepurchase',
    'debt',
    'debtRepayment'
]

finance_df = finance_df.drop(columns=columns_to_drop)

# Imputar
finance_df = finance_df.fillna(0)

#Metricas (tarjetas) -----------------------------------------------------------------------------------------------

 #12 PRINCIPIOS DE BUFFETT

#1. ROE ≥ 15%
#2. Deuda baja (D/E < 0.5)
#3. Margen neto > 20%
#4. P/E razonable (< 20x)
#5. Crecimiento EPS ≥ 10%/año
#6. Ganancias consistentes 10 años (sin pérdidas anuales)
#7. Margen bruto > 40% (poder de fijación de precios = moat)
#8. Current ratio > 1.5
#9. Ventaja competitiva durable (moat cualitativo)
#10. Negocio entendible en menos de 10 minutos
#11. Management de calidad con historial probado
#12. Crecimiento de ingresos ≥ 7%/año

#Ultimos 10 años

principios_df = finance_df[
    ['fiscalDateEnding','ROE%','netIncomeMargin%','b_currentLongTermDebt','b_capitalLeaseObligations','b_longTermDebt','b_totalShareholderEquity',
     'i_netIncome','i_totalRevenue','i_grossProfit','b_totalCurrentAssets','b_totalCurrentLiabilities','i_reportedEPS','ROIC%']
].copy()

principios_df['fiscalDateEnding'] = pd.to_datetime(principios_df['fiscalDateEnding'])

principios_df = principios_df.sort_values('fiscalDateEnding')

#Precio actual: viene de fuera (ejemplo con yfinance) ---

precio_actual = (yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]).round(2)

precio_actual_tarjeta = f"${yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1].round(2)} USD"


# 0. ROIC
p0_ROIC = principios_df.iloc[-1]['ROIC%'].round(1)
p0_ROIC_tarjeta = f"{p0_ROIC}%"

# 1. ROE >= 15%: ROE%
p1_ROE = principios_df.iloc[-1]['ROE%'].round(1)
p1_ROE_tarjeta = f"{p1_ROE}%"

# 2. Deuda baja (D/E < 0.5): debt_to_equity
principios_df['deuda_total'] = principios_df["b_currentLongTermDebt"] + principios_df["b_capitalLeaseObligations"] + principios_df["b_longTermDebt"]
principios_df['debt_to_equity'] = principios_df['deuda_total']  / principios_df["b_totalShareholderEquity"]

p2_deuda = principios_df.iloc[-1]['debt_to_equity'].round(2)
p2_deuda_tarjeta = f"{p2_deuda}x"

# 3. Margen neto > 20%:netIncomeMargin%
p3_margen_neto = principios_df.iloc[-1]['netIncomeMargin%'].round(1)
p3_margen_neto_tarjeta = f"{p3_margen_neto}%"

# 4. Current ratio > 1.5
principios_df['razon_circulante'] = principios_df['b_totalCurrentAssets'] / principios_df['b_totalCurrentLiabilities']
p4_razon_circulante = principios_df.iloc[-1]['razon_circulante'].round(2)
p4_razon_circulante_tarjeta = f"{p4_razon_circulante}x"

# 5. Crecimiento EPS >= 10%/año -> CAGR 5 años (mismo que ⑥.6)
latest = principios_df.iloc[-1]      # año fiscal más reciente
prev5  = principios_df.iloc[-6]      # 5 años atrás (para CAGR)

principios_df['crecimiento_eps'] = ((latest["i_reportedEPS"] / prev5["i_reportedEPS"]) ** (1/5) - 1) * 100

p5_crecimiento_eps = principios_df.iloc[-1]['crecimiento_eps'].round(1)
p5_crecimiento_eps_tarjeta = f"{p5_crecimiento_eps}%"


# 6. Ganancias consistentes 10 años (sin pérdidas anuales) (i_netIncome positivo)
ultimos10 = principios_df.iloc[-10:]

p6_ganancias_consistentes = np.where(
    (ultimos10['i_netIncome'] > 0).all(),
    "Ganancia Consistente",
    "Pérdida"
)

# 7. Margen bruto > 40%
principios_df['grossProfit%'] = (principios_df['i_grossProfit']/ principios_df['i_totalRevenue'])*100
p7_margen_bruto = principios_df.iloc[-1]['grossProfit%'].round(1)
p7_margen_bruto_tarjeta = f"{p7_margen_bruto}%"


# 8. P/E razonable (< 20x) -> requiere precio externo
fecha_reporte = latest['fiscalDateEnding'] #Fecha del último año fiscal reportado

# Precio de cierre del último día hábil disponible en torno al año fiscal
precio_historico = (
    yf.Ticker(ticker)
    .history(
        start=fecha_reporte - pd.Timedelta(days=7),
        end=fecha_reporte + pd.Timedelta(days=1)
    )["Close"]
)

if precio_historico.empty:
    st.warning(
        f"No se encontraron precios cercanos a la fecha de cierre fiscal "
        f"({fecha_reporte.date()}). Se usará el precio actual como referencia."
    )
    precio_ultimo_año = precio_actual
else:
    precio_ultimo_año = precio_historico.iloc[-1].round(2)

# EPS del último año fiscal reportado
eps_ultimo_año = latest["i_reportedEPS"]

# P/E
p8_pe_ratio = (precio_ultimo_año / eps_ultimo_año).round(1)
p8_pe_ratio_tarjeta = f"{p8_pe_ratio}x"


# 9. Moat cualitativo -> no calculable desde datos numéricos, input manual
# 10. Negocio entendible en <10 min -> input manual
# 11. Management de calidad -> input manual

# 12. Crecimiento de ingresos >= 7%/año (CAGR 5 años, mismo criterio que EPS)
principios_df['crecimiento_ingresos_5a'] = ((latest["i_totalRevenue"] / prev5["i_totalRevenue"]) ** (1/5) - 1) * 100
p12_crecimiento_ingresos = principios_df.iloc[-1]['crecimiento_ingresos_5a'].round(1)
p12_crecimiento_ingresos_tarjeta = f"{p12_crecimiento_ingresos}%"

VERDE  = '<span style="color:#32CD32; font-size:18px;">●</span>'
AMARILLO = '<span style="color:#EFB810; font-size:18px;">●</span>'
ROJO   = '<span style="color:#FF3B30; font-size:18px;">●</span>'

# ROE -> verde >=15% | amarillo 10-14% | rojo <10%
def color_roe(x):
    if x >= 15:
        return VERDE
    elif x >= 10:
        return AMARILLO
    else:
        return ROJO

p1_color = color_roe(p1_ROE)

#Deuda/Capital → verde <0.5x | amarillo 0.5–1.5x | rojo >1.5x
def color_deuda(x):
    if x < 0.50:
        return VERDE
    elif x <= 1.50:
        return AMARILLO
    else:
        return ROJO

p2_color = color_deuda(p2_deuda)

#Margen neto → verde >20% | amarillo 10–20% | rojo <10%
def color_margen_neto(x):
    if x > 20:
        return VERDE
    elif x >= 10:
        return AMARILLO
    else:
        return ROJO

p3_color = color_margen_neto(p3_margen_neto)

def color_current_ratio(x):
    if x > 1.5:
        return VERDE
    elif x >= 1.0:
        return AMARILLO
    else:
        return ROJO

p4_color = color_current_ratio(p4_razon_circulante)

def color_eps(x):
    if x >= 10:
        return VERDE
    elif x >= 5:
        return AMARILLO
    else:
        return ROJO

p5_color = color_eps(p5_crecimiento_eps)

if p6_ganancias_consistentes == "Ganancia Consistente":
    p6_color = VERDE
else:
    p6_color = ROJO

def color_margen_bruto(x):
    if x > 40:
        return VERDE
    elif x >= 25:
        return AMARILLO
    else:
        return ROJO

p7_color = color_margen_bruto(p7_margen_bruto)

#P/E ratio   → verde <15x | amarillo 15–20x | rojo >20x
def color_pe_ratio(x):
    if x < 15:
        return VERDE
    elif x <= 20:
        return AMARILLO
    else:
        return ROJO

p8_color = color_pe_ratio(p8_pe_ratio)

#Crecim. EPS 5a   → verde ≥10% | amarillo 5–9% | rojo <5%

def color_ingresos(x):
    if x >= 10:
        return VERDE
    elif x >= 5:
        return AMARILLO
    else:
        return ROJO

p12_color = color_ingresos(p12_crecimiento_ingresos)



#MODELO____________________________________________________________________________________________________________________
def obtener_precios_yahoo(ticker):
    inicio = "1999-12-01"
    fin = datetime.today().strftime("%Y-%m-%d")

    price_df = yf.download(
        ticker,
        start=inicio,
        end=fin,
        interval="1mo",
        auto_adjust=False,
        progress=False
    )

    # Si yfinance devuelve MultiIndex
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)

    price_df = price_df.reset_index()

    # Solo fecha y precio ajustado
    price_df = price_df[["Date", "Adj Close"]]

    # Renombrar columna
    price_df = price_df.rename(columns={"Adj Close": "adjusted_close"})

    # Asegurar que el precio sea numérico
    price_df["adjusted_close"] = pd.to_numeric(
        price_df["adjusted_close"],
        errors="coerce"
    )

    return price_df


price_df = obtener_precios_yahoo(ticker)


# Establecer indice
price_df = price_df.set_index(price_df.columns[0]) 
price_df.index = pd.to_datetime(price_df.index)
price_df.sort_index(inplace=True)
price_df.columns.name = None


# Re-muestrear para obtener el último precio de cierre ajustado de cada año (cierre de año).
price_df = price_df.resample('YE')['adjusted_close'].last().to_frame()

last_price = int(price_df['adjusted_close'].iloc[-1])

finance_df['netDebt'] = finance_df['b_shortTermDebt']+finance_df['b_longTermDebt']-finance_df['b_cashAndShortTermInvestments']-finance_df['b_shortTermInvestments']
finance_df['netDebt/EBITDA'] = finance_df['netDebt']/finance_df['c_ebitda']
finance_df['marketCapitalization'] = finance_df['b_commonStockSharesOutstanding']*last_price
finance_df['enterpriseValue'] = finance_df['marketCapitalization'] + finance_df['netDebt']

# Multiplos:
finance_df['PER'] = finance_df['marketCapitalization']/finance_df['i_netIncome']
finance_df['EV/FCF'] = finance_df['enterpriseValue']/finance_df['freeCashFlow']
finance_df['EV/EBITDA'] = finance_df['enterpriseValue']/finance_df['c_ebitda']
finance_df['EV/EBIT'] = finance_df['enterpriseValue']/finance_df['i_operatingIncome']

#finance_df[['PER','EV/FCF','EV/EBITDA','EV/EBIT']].loc[0]

per_multiple = int(finance_df['PER'].iloc[0])
ev_fcf_multiple = int(finance_df['EV/FCF'].iloc[0])
ev_ebitda_multiple = int(finance_df['EV/EBITDA'].iloc[0])
ev_ebit_multiple = int(finance_df['EV/EBIT'].iloc[0])

#per_multiple, ev_fcf_multiple, ev_ebitda_multiple, ev_ebit_multiple
#last_price

#Modelado

forecast_df = finance_df[['fiscalDateEnding','b_commonStockSharesOutstanding','netDebt','freeCashFlow','i_operatingIncome','i_netIncome','c_ebitda']].copy()
forecast_df['fiscalDateEnding'] = pd.to_datetime(forecast_df['fiscalDateEnding'], errors='coerce')
forecast_df = forecast_df.sort_values(by='fiscalDateEnding').reset_index(drop=True)
numerical_cols = ['b_commonStockSharesOutstanding','netDebt','freeCashFlow','i_operatingIncome','i_netIncome','c_ebitda']
last_historical_values = forecast_df[numerical_cols].iloc[-1]

last_year = forecast_df['fiscalDateEnding'].dt.year.max()
future_years = pd.to_datetime([f"{last_year + i}-12-31" for i in range(1, 6)])

future_df = pd.DataFrame({'fiscalDateEnding': future_years})

for col in numerical_cols:
    future_df[col] = last_historical_values[col]

extended_forecast_df = pd.concat([forecast_df, future_df], ignore_index=True)
extended_forecast_df = extended_forecast_df.sort_values(by='fiscalDateEnding').reset_index(drop=True)

numerical_cols_extended = extended_forecast_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
valuation_df = pd.DataFrame({'fiscalDateEnding': extended_forecast_df['fiscalDateEnding']})

for col in numerical_cols_extended:
    valuation_df[f'EMA_5Y_{col}'] = extended_forecast_df[col].ewm(span=5, adjust=False).mean()

#Valuacion:
valuation_df['priceEV/FCF'] = ((valuation_df['EMA_5Y_freeCashFlow'] * ev_fcf_multiple) - valuation_df['EMA_5Y_netDebt']) / valuation_df['EMA_5Y_b_commonStockSharesOutstanding']
valuation_df['priceEV/EBITDA'] = ((valuation_df['EMA_5Y_c_ebitda'] * ev_ebitda_multiple) - valuation_df['EMA_5Y_netDebt']) / valuation_df['EMA_5Y_b_commonStockSharesOutstanding']
valuation_df['priceEV/EBIT'] = ((valuation_df['EMA_5Y_i_operatingIncome'] * ev_ebit_multiple) - valuation_df['EMA_5Y_netDebt']) / valuation_df['EMA_5Y_b_commonStockSharesOutstanding']

valuation_df['pricePERexCash'] = np.where(
    valuation_df['EMA_5Y_netDebt'] < 0,
    ((valuation_df['EMA_5Y_i_netIncome'] * per_multiple) - valuation_df['EMA_5Y_netDebt']) / valuation_df['EMA_5Y_b_commonStockSharesOutstanding'],
    (valuation_df['EMA_5Y_i_netIncome'] * per_multiple) / valuation_df['EMA_5Y_b_commonStockSharesOutstanding']
)

intrinsic_value = valuation_df[['fiscalDateEnding','priceEV/FCF','priceEV/EBITDA','priceEV/EBIT','pricePERexCash']].tail()

intrinsic_value.set_index('fiscalDateEnding', inplace=True)

#intrinsic_value

valor_intrinseco_tarjeta = f"${intrinsic_value.iloc[-1].median().round(2)} USD"

profitability = np.round((intrinsic_value.iloc[-1].median()*100/last_price)**(1/5)-1, 2)
#profitability


# Tabla estructurada ________________________________________________________________________________________________

tabla_estructurada = finance_df[
    ['fiscalDateEnding',
    'ROA%',
    'ROE%',
    'ROIC%',
    'reinvestmentRate%',
    'buybacks%',
    'i_totalRevenue',
    'i_totalRevenue_YoY_growth%',
    'i_grossProfit',
    'i_costOfRevenue',
    'i_ebitda',
    'ebitdaMargin%',
    'i_ebitda_YoY_growth%',
    'ebitMargin%',
    'i_operatingIncome',
    'i_operatingIncome_YoY_growth%',
    'i_interestExpense',
    'i_interestIncome',
    'netInterest',
    'i_incomeBeforeTax',
    'i_incomeTaxExpense',
    'taxRate%',
    'i_netIncome_YoY_growth%',
    'i_reportedEPS',
    'i_reportedEPS_YoY_growth%',
    'i_netIncome',
    'netIncomeMargin%',
    'b_commonStockSharesOutstanding',
    'b_commonStockSharesOutstanding_YoY_growth%',
    'b_inventory',
    'b_currentNetReceivables',
    'b_currentAccountsPayable',
    'b_deferredRevenue',
    'b_cashAndShortTermInvestments',
    'b_shortTermInvestments',
    'b_shortTermDebt',
    'b_longTermDebt',
    'b_capitalLeaseObligations',
    'b_totalShareholderEquity',
    'freeCashFlow',
    'freeCashFlow_YoY_growth%',
    'freeCashFlowPerShare',
    'freeCashFlowPerShare_YoY_growth%',
    'maintenanceCapexMargin%',
    'workingCapitalMargin%',
    'freeCashFlowMargin%',
    'cashConversion%',
    'c_ebitda',
    'c_capitalExpenditures',
    'maintenanceCapex',
    'growthCapex',
    'workingCapital',
    'changeInWC',
    'investedCapital',
    'c_depreciationDepletionAndAmortization',
    'organicGrowth%',
    'dividendPayout%',
    'debtAmortization%',
    'total%'
     ]
].copy()

#Convertir columnas indicadas a millones

columns_in_millions_income = [
    'i_totalRevenue',
    'i_grossProfit',
    'i_costOfRevenue',
    'i_ebitda',
    'i_operatingIncome',
    'i_interestExpense',
    'i_interestIncome',
    'netInterest',
    'i_incomeBeforeTax',
    'i_incomeTaxExpense',
    'i_netIncome',
    'b_commonStockSharesOutstanding',
    'b_inventory',
    'b_currentNetReceivables',
    'b_currentAccountsPayable',
    'b_deferredRevenue',
    'b_cashAndShortTermInvestments',
    'b_shortTermInvestments',
    'b_shortTermDebt',
    'b_longTermDebt',
    'b_capitalLeaseObligations',
    'b_totalShareholderEquity',
    'freeCashFlow',
    'c_ebitda',
    'c_capitalExpenditures',
    'maintenanceCapex',
    'growthCapex',
    'workingCapital',
    'changeInWC',
    'investedCapital',
    'c_depreciationDepletionAndAmortization'
]

tabla_estructurada['fiscalDateEnding'] = tabla_estructurada['fiscalDateEnding'].dt.year
tabla_estructurada = convert_columns_to_millions(tabla_estructurada.copy(), columns_in_millions_income)

mapping = {
    'fiscalDateEnding':f'{ticker}',
    'ROA%': 'ROA %',
    'ROE%': 'ROE %',
    'ROIC%': 'ROIC %',
    'reinvestmentRate%': 'Tasa de reinversión %',
    'buybacks%': 'Recompras de acciones %',
    'i_totalRevenue': 'Ventas',
    'i_totalRevenue_YoY_growth%': 'Crecimiento de ventas %',
    'i_grossProfit': 'Utilidad bruta',
    'i_costOfRevenue': 'Costo de ventas',
    'i_ebitda': 'EBITDA',
    'ebitdaMargin%': 'Margen EBITDA %',
    'i_ebitda_YoY_growth%': 'Crecimiento EBITDA %',
    'ebitMargin%': 'Margen EBIT %',
    'i_operatingIncome': 'Utilidad operativa (EBIT)',
    'i_operatingIncome_YoY_growth%': 'Crecimiento EBIT %',
    'i_interestExpense': 'Gastos por intereses',
    'i_interestIncome': 'Ingreso por intereses',
    'netInterest': 'Interés neto',
    'i_incomeBeforeTax': 'Utilidad antes de impuestos',
    'i_incomeTaxExpense': 'Impuesto sobre la renta',
    'taxRate%': 'Tasa impositiva %',
    'i_netIncome_YoY_growth%': 'Crecimiento de utilidad neta %',
    'i_reportedEPS': 'Utilidad por acción (EPS)',
    'i_reportedEPS_YoY_growth%': 'Crecimiento EPS %',
    'i_netIncome': 'Utilidad neta',
    'netIncomeMargin%': 'Margen neto %',
    'b_commonStockSharesOutstanding': 'Acciones en circulación',
    'b_commonStockSharesOutstanding_YoY_growth%': 'Crecimiento de acciones en circulación %',
    'b_inventory': 'Inventarios',
    'b_currentNetReceivables': 'Cuentas por cobrar',
    'b_currentAccountsPayable': 'Cuentas por pagar',
    'b_deferredRevenue': 'Ingresos diferidos',
    'b_cashAndShortTermInvestments': 'Efectivo e inversiones de corto plazo',
    'b_shortTermInvestments': 'Inversiones de corto plazo',
    'b_shortTermDebt': 'Deuda a corto plazo',
    'b_longTermDebt': 'Deuda a largo plazo',
    'b_capitalLeaseObligations': 'Obligaciones por arrendamiento',
    'b_totalShareholderEquity': 'Capital contable',
    'freeCashFlow': 'Flujo de caja libre',
    'freeCashFlow_YoY_growth%': 'Crecimiento del flujo de caja libre %',
    'freeCashFlowPerShare': 'Flujo de caja libre por acción',
    'freeCashFlowPerShare_YoY_growth%': 'Crecimiento del FCF por acción %',
    'maintenanceCapexMargin%': 'Costo de mantenimiento %',
    'workingCapitalMargin%': 'Costo de capital de trabajo %',
    'freeCashFlowMargin%': 'Margen de flujo de caja libre %',
    'cashConversion%': 'Conversión de efectivo %',
    'c_ebitda': 'EBITDA estimado',
    'c_capitalExpenditures': 'Gastos de capital (CapEx)',
    'maintenanceCapex': 'CapEx de mantenimiento',
    'growthCapex': 'CapEx de crecimiento',
    'workingCapital': 'Capital de trabajo',
    'changeInWC': 'Cambio en capital de trabajo',
    'investedCapital': 'Capital invertido',
    'c_depreciationDepletionAndAmortization': 'Depreciación y amortización',
    'organicGrowth%': 'Crecimiento orgánico %',
    'dividendPayout%': 'Pago de dividendos %',
    'debtAmortization%': 'Amortización de deuda %',
    'total%': 'Puntuación total %'
}

tabla_estructurada = tabla_estructurada.rename(columns=mapping)

# Sort by Ticker in ascending order before transposing
tabla_vertical = tabla_estructurada.sort_values(by=ticker).set_index(ticker).T

tabla_vertical = round_numeric_df1miles(tabla_vertical)

# ------------------------------------------------------------------
# Diseño TABLA_VERTICAL CON CATEGORÍAS DE COLOR
# ------------------------------------------------------------------

CATEGORY_COLORS = {
    "Estado de resultados": "#8B5CF6",  # morado
    "Balance general": "#10B981",       # verde
    "Flujo de efectivo": "#F97316",     # naranja
    "Ratios": "#3B82F6",      # azul
}

METRIC_CATEGORY = {
    # Ratios
    "ROA %": "Ratios",
    "ROE %": "Ratios",
    "ROIC %": "Ratios",
    "Tasa de reinversión %": "Ratios",
    "Recompras de acciones %": "Ratios",

    # Estado de resultados
    "Ventas": "Estado de resultados",
    "Crecimiento de ventas %": "Estado de resultados",
    "Utilidad bruta": "Estado de resultados",
    "Costo de ventas": "Estado de resultados",
    "EBITDA": "Estado de resultados",
    "Margen EBITDA %": "Estado de resultados",
    "Crecimiento EBITDA %": "Estado de resultados",
    "Margen EBIT %": "Estado de resultados",
    "Utilidad operativa (EBIT)": "Estado de resultados",
    "Crecimiento EBIT %": "Estado de resultados",
    "Gastos por intereses": "Estado de resultados",
    "Ingreso por intereses": "Estado de resultados",
    "Interés neto": "Estado de resultados",
    "Utilidad antes de impuestos": "Estado de resultados",
    "Impuesto sobre la renta": "Estado de resultados",
    "Tasa impositiva %": "Estado de resultados",
    "Crecimiento de utilidad neta %": "Estado de resultados",
    "Utilidad por acción (EPS)": "Estado de resultados",
    "Crecimiento EPS %": "Estado de resultados",
    "Utilidad neta": "Estado de resultados",
    "Margen neto %": "Estado de resultados",

    # Balance general
    "Acciones en circulación": "Balance general",
    "Crecimiento de acciones en circulación%": "Balance general",
    "Inventarios": "Balance general",
    "Cuentas por cobrar": "Balance general",
    "Cuentas por pagar": "Balance general",
    "Ingresos diferidos": "Balance general",
    "Efectivo e inversiones de corto plazo": "Balance general",
    "Inversiones de corto plazo": "Balance general",
    "Deuda a corto plazo": "Balance general",
    "Deuda a largo plazo": "Balance general",
    "Obligaciones por arrendamiento": "Balance general",
    "Capital contable": "Balance general",

    # Flujo de efectivo
    "Flujo de caja libre": "Flujo de efectivo",
    "Crecimiento del flujo de caja libre %": "Flujo de efectivo",
    "Flujo de caja libre por acción": "Flujo de efectivo",
    "Crecimiento del FCF por acción %": "Flujo de efectivo",
    "Costo de mantenimiento %": "Flujo de efectivo",
    "Costo de capital de trabajo %": "Flujo de efectivo",
    "Margen de flujo de caja libre %": "Flujo de efectivo",
    "Conversión de efectivo %": "Flujo de efectivo",
    "EBITDA estimado": "Flujo de efectivo",
    "Gastos de capital (CapEx)": "Flujo de efectivo",
    "CapEx de mantenimiento": "Flujo de efectivo",
    "CapEx de crecimiento": "Flujo de efectivo",
    "Capital de trabajo": "Flujo de efectivo",
    "Cambio en capital de trabajo": "Flujo de efectivo",
    "Capital invertido": "Flujo de efectivo",
    "Depreciación y amortización": "Flujo de efectivo",
    "Crecimiento orgánico %": "Flujo de efectivo",
    "Pago de dividendos %": "Flujo de efectivo",
    "Amortización de deuda %": "Flujo de efectivo",
    "Puntuación total %": "Flujo de efectivo",
}

DEFAULT_COLOR = "#9CA3AF"  # gris, por si alguna métrica no está mapeada



def _tv_css() -> str:
    return """
    <style>
    .tv-card {
        background: white;
        border-radius: 14px;
        padding: 20px 24px;
        border: 1px solid #eee;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        font-family: Arial, Helvetica, sans-serif;
    }
    .tv-scroll {
        overflow-x: scroll;
        width: 100%;
    }
    .tv-scroll::-webkit-scrollbar {
        height: 12px;
    }
    .tv-scroll::-webkit-scrollbar-track {
        background: #F5F3FE;
        border-radius: 10px;
    }
    .tv-scroll::-webkit-scrollbar-thumb {
        background: #C9BFF7;
        border-radius: 10px;
    }
    .tv-scroll::-webkit-scrollbar-thumb:hover {
        background: #A99AE8;
    }
    .tv-title { font-size: 18px; font-weight: 800; color: #1F1F1F; margin-bottom: 14px; }

    table.tv-table { width: 100%; min-width: 1400px; border-collapse: collapse; font-size: 13px; }

    .tv-legend { margin: 6px 0 18px 0; display: flex; flex-wrap: wrap; gap: 10px; }
    .tv-chip {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 700;
    }
    .tv-chip-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

    table.tv-table { width: 100%; border-collapse: collapse; font-size: 13px; }

    table.tv-table th, table.tv-table td {
        border: none;
    }
    table.tv-table th {
        text-align: right; color: #6B7280; font-weight: 700; padding: 8px 10px;
        border-bottom: 1px solid #E4DFFB; white-space: nowrap; position: sticky; top: 0;
        background: #F5F3FE;
    }
    table.tv-table th:first-child { text-align: left; position: sticky; left: 0; z-index: 2; background: #F5F3FE; }
    table.tv-table td {
        text-align: right; padding: 9px 10px; border-bottom: 1px solid #EDE9FC;
        color: #222; white-space: nowrap;
    }

    table.tv-table td.tv-cell {
        cursor: pointer;
        outline: none;
    }
    table.tv-table td.tv-cell:focus,
    table.tv-table td.tv-cell.tv-selected {
        box-shadow: inset 0 0 0 2px #7C6CF5;
        background-color: #F0EDFE;
    }

    table.tv-table td.tv-metric-cell {
        text-align: left; font-weight: 600; position: sticky; left: 0; background: white;
        padding-left: 16px; z-index: 1;
    }
    table.tv-table td.tv-metric-cell .tv-bar {
        position: absolute; left: 0; top: 6px; bottom: 6px; width: 3px; border-radius: 2px;
    }
    table.tv-table td.tv-metric-cell .tv-dot  {
        display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px;
    }
    </style>
    """


def render_tabla_vertical(tabla_vertical, titulo: str = ""):
    st.markdown(_tv_css(), unsafe_allow_html=True)

    html = ['<div class="tv-card">']

    if titulo:
        html.append(f'<div class="tv-title">{titulo}</div>')

    # Leyenda con puntos de color
    html.append('<div class="tv-legend">')
    for cat, color in CATEGORY_COLORS.items():
        html.append(
            f'<span class="tv-chip" style="background:{color}22;color:{color};">'
            f'<span class="tv-chip-dot" style="background:{color};"></span>{cat}</span>'
        )
    html.append("</div>")


    # Tabla
    html.append('<div class="tv-scroll">')
    html.append('<table class="tv-table"><thead><tr><th>Métrica</th>')

    for col in tabla_vertical.columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead><tbody>")



    for r_idx, (metric_name, row) in enumerate(tabla_vertical.iterrows()):
        cat = METRIC_CATEGORY.get(metric_name)
        color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        html.append("<tr>")
        html.append(
            f'<td class="tv-metric-cell">'
            f'<span class="tv-bar" style="background:{color};"></span>'
            f'<span class="tv-dot" style="background:{color};"></span>'
            f"{metric_name}</td>"
        )
        for c_idx, val in enumerate(row):
            html.append(f'<td class="tv-cell" tabindex="0" data-row="{r_idx}" data-col="{c_idx}">{val}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div></div>")


    st.markdown("".join(html), unsafe_allow_html=True)


# diseño
def _chart_css() -> str:
    return textwrap.dedent("""
    <style>
    .chart-card {
        background: #F7F7F9;
        border-radius: 18px;
        padding: 14px 18px 12px 18px;
        border: 1px solid #eee;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        font-family: Arial, Helvetica, sans-serif;
        margin-bottom: 14px;
    }
    .chart-title { font-size: 15px; font-weight: 800; color:#1F1F1F; margin-bottom:2px; }
    .chart-subtitle { font-size: 12px; color:#9A9AA5; margin-bottom:10px; }

    .bars-scroll {
        overflow-x: auto;
        width: 100%;
        padding-bottom: 6px;
    }
    .bars-scroll::-webkit-scrollbar {
        height: 10px;
    }
    .bars-scroll::-webkit-scrollbar-track {
        background: #F5F3FE;
        border-radius: 10px;
    }
    .bars-scroll::-webkit-scrollbar-thumb {
        background: #C9BFF7;
        border-radius: 10px;
    }
    .bars-scroll::-webkit-scrollbar-thumb:hover {
        background: #A99AE8;
    }
    .bars-row {
        display:flex;
        align-items:flex-end;
        gap:12px;
        min-width: max-content;
    }
    .bar-wrap {
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:flex-end;
        flex: 0 0 auto;
        width: 56px;
        height:100%;
    }

    .bar-value {
        font-size: 11px;
        font-weight:700;
        color:#4B3AD6;
        margin-bottom:6px;
        white-space:nowrap;
    }
    .bar-purple {
        width:100%;
        max-width:46px;
        border-radius: 10px 10px 4px 4px;
        background: linear-gradient(180deg, #C9BFF7 0%, #7C6CF5 55%, #4B3AD6 100%);
        transition: height 0.2s ease;
    }
    .bar-label {
        font-size:11px;
        color:#4A4A4A;
        margin-top:8px;
        white-space:nowrap;
    }

    .bar-up {
        width:100%;
        max-width:46px;
        border-radius: 10px 10px 4px 4px;
        background: linear-gradient(180deg, #C9BFF7 0%, #7C6CF5 55%, #4B3AD6 100%);
        transition: height 0.2s ease;
    }
    .bar-value-up { color:#4B3AD6; }

    .bar-down {
        width:100%;
        max-width:46px;
        border-radius: 10px 10px 4px 4px;
        background: linear-gradient(180deg, #E0B8DB 0%, #B565A7 55%, #8B3A78 100%);
        transition: height 0.2s ease;
    }
    .bar-value-down { color:#8B3A78; }

    </style>
    """)


def render_bar_card_html(labels, values, titulo: str, subtitulo: str = "", sufijo: str = "%", prefijo: str = "", max_height: int = 80, bar_width: int = 36, label_step: int = 1):
    valores_num = [float(v) if pd.notna(v) else 0.0 for v in values]
    max_val = max([abs(v) for v in valores_num]) if valores_num else 1
    max_val = max_val if max_val != 0 else 1

    bars_html = []
    for i, (label, val) in enumerate(zip(labels, valores_num)):
        height_px = max(int((abs(val) / max_val) * max_height), 3)
        valor_formateado = f"{prefijo}{val:,.0f}{sufijo}" if prefijo else f"{val:.1f}{sufijo}"
        valor_mostrado = valor_formateado if i % label_step == 0 else "&nbsp;"

        # Comparación con el valor anterior para determinar color/flecha
        if i == 0:
            bar_class = "bar-purple"
            value_class = ""
            flecha = ""
        else:
            anterior = valores_num[i - 1]
            if val > anterior:
                bar_class = "bar-up"
                value_class = "bar-value-up"
                flecha = "▲ "
            elif val < anterior:
                bar_class = "bar-down"
                value_class = "bar-value-down"
                flecha = "▼ "
            else:
                bar_class = "bar-purple"
                value_class = ""
                flecha = ""

        valor_con_flecha = f"{flecha}{valor_mostrado}" if valor_mostrado != "&nbsp;" else valor_mostrado

        bars_html.append(
            f'<div class="bar-wrap" style="width:{bar_width}px;">'
            f'<div class="bar-value {value_class}">{valor_con_flecha}</div>'
            f'<div class="{bar_class}" style="height:{height_px}px;"></div>'
            f'<div class="bar-label">{label}</div>'
            f'</div>'
        )

    html = (
        f'<div class="chart-card">'
        f'<div class="chart-title">{titulo}</div>'
        f'<div class="bars-scroll"><div class="bars-row">{"".join(bars_html)}</div></div>'
        f'</div>'
    )

    st.markdown(_chart_css() + html, unsafe_allow_html=True)



# ------------------------------------------------------------------
# CONTENIDO PRINCIPAL
# ------------------------------------------------------------------
#Titulo del nombre de la corporacion =======================
info = yf.Ticker(ticker).info
#corporation_name = info.get("longName", ticker).upper()
#st.markdown(f"## **{corporation_name}**")

corporation_name = info.get("longName", ticker).upper()

st.markdown(
    f"""
    <div style="
        font-family: Arial, Helvetica, sans-serif;
        font-size: 32px;
        font-weight: 900;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #1F1F1F;
        margin-bottom: 1px;
    ">
        {corporation_name}
    </div>
    """,
    unsafe_allow_html=True
)
#info["longName"]          # Microsoft Corporation
sector = info["sector"]             # Technology
industry = info["industry"]           # Software - Infrastructure

st.markdown(
    f"""
    <div style="
        font-family: Arial, Helvetica, sans-serif;
        font-size: 14px;
        font-weight: 500;
        color: #808080;
        margin-top: 0px;
        margin-bottom: 20px;
    ">
        {ticker} &nbsp;•&nbsp; {sector} &nbsp;•&nbsp; {industry}
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------
# CONTENIDO POR SECCIÓN DEL MENU
# ------------------------------------------------------------------
current = st.session_state.menu_selection
labels = {item_id: label for item_id, label in MENU_ITEMS}
 
if current == "dashboard":

    st.markdown(
        f"""
        <div class="cards-row">
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon"><i class="bi bi-graph-up-arrow"></i></div>
                    <div class="card-label">Precio actual</div>
                </div>
                <div class="card-value">{precio_actual_tarjeta}</div>
            </div>
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon"><i class="bi bi-pie-chart"></i></div>
                    <div class="card-label">Valor intrínseco</div>
                </div>
                <div class="card-value">{valor_intrinseco_tarjeta}</div>
            </div>
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon"><i class="bi-trophy"></i></div>
                    <div class="card-label">EV/FCF</div>
                </div>
                <div class="card-value">{ev_fcf_multiple}x</div>
            </div>
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon"><i class="bi bi-shield-check"></i></div>
                    <div class="card-label">EV/EBITDA</div>
                </div>
                <div class="card-value">{ev_ebitda_multiple}x</div>
            </div>
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon"><i class="bi bi-shield-check"></i></div>
                    <div class="card-label">EV/EBIT</div>
                </div>
                <div class="card-value">{ev_ebit_multiple}x</div>
            </div>
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon"><i class="bi bi-shield-check"></i></div>
                    <div class="card-label">PER</div>
                </div>
                <div class="card-value">{per_multiple}x</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # graficos
    metricas_dashboard = {
        "Margen neto %": "Margen neto %",
        "Margen EBIT %": "Margen EBIT %",
        "Margen EBITDA %": "Margen EBITDA %",
        "ROA": "ROA %",
        "ROE": "ROE %",
        "ROIC": "ROIC %",
    }

    metrica_elegida = st.selectbox(
        "Selecciona una métrica",
        options=list(metricas_dashboard.keys()),
        key="metrica_dashboard_selector",
    )

    metrica_col = metricas_dashboard[metrica_elegida]

    if metrica_col not in tabla_vertical.index:
        st.warning(f"No se encontró la métrica '{metrica_col}' en la tabla.")
    else:
        serie = tabla_vertical.loc[metrica_col].astype(str).str.replace(',', '', regex=False)
        serie_num = pd.to_numeric(serie, errors='coerce')
        anios = [str(a) for a in tabla_vertical.loc[metrica_col].index]

        render_bar_card_html(
            labels=anios,
            values=serie_num.values,
            titulo="",
            sufijo="%",
            prefijo="",
            label_step=1,
        )


    # Tarjetas 
    st.markdown(
        f"""
        <div class="wide-card-row">
            <div class="info-card-wide">
                <div class="card-label" style="color:#7C6CF5;">Principios</div>
                <div style="display:flex; gap:24px;">
                    <ol class="card-desc-list" style="flex:1; margin:0; padding-left:18px;">
                        <li>{p1_color} ROE ≥ 15%:  <b>{p1_ROE_tarjeta}</b></li>
                        <li>{p3_color} Margen neto > 20%:  <b>{p3_margen_neto_tarjeta}</b></li>
                        <li>{p7_color} Margen bruto > 40%:  <b>{p7_margen_bruto_tarjeta}</b></li>
                    </ol>
                    <ol class="card-desc-list" style="flex:1; margin:0; padding-left:18px;" start="4">
                        <li>{p5_color} Crecimiento EPS ≥ 10%/año (5Y):  <b>{p5_crecimiento_eps_tarjeta}</b></li>
                        <li>{p12_color} Crecimiento de ingresos ≥ 7%/año (5Y):  <b>{p12_crecimiento_ingresos_tarjeta}</b></li>
                        <li>{p2_color} Deuda baja < 0.5:  <b>{p2_deuda_tarjeta}</b></li>
                    </ol>
                    <ol class="card-desc-list" style="flex:1; margin:0; padding-left:18px;" start="7">
                        <li>{p4_color} Razón circulante > 1.5:  <b>{p4_razon_circulante_tarjeta}</b></li>
                        <li>{p8_color} P/E razonable < 20x:  <b>{p8_pe_ratio_tarjeta}</b></li>
                        <li>{p6_color} Ganancias consistentes (10 años)</li>
                    </ol>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


elif current == "Finanzas clave":
    render_tabla_vertical(
        tabla_vertical,
        titulo="Panel financiero (USD)",
    )
 
elif current == "graficos":
 
    def load_income_sankey_data(ticker):
        income_df_trim = st.session_state["income_df_trim"].copy()
        income_df_trim['fiscalDateEnding'] = pd.to_datetime(income_df_trim['fiscalDateEnding'])

        df = income_df_trim[
            ['fiscalDateEnding','totalRevenue','costOfRevenue','netIncome','operatingExpenses',
            'operatingIncome','grossProfit','researchAndDevelopment',
            'sellingGeneralAndAdministrative','incomeTaxExpense','incomeBeforeTax']
        ].copy()

        df = convert_columns_to_numeric(df, columns_to_exclude=['fiscalDateEnding'])
        df = df[df['fiscalDateEnding'].dt.year >= 2024].sort_values('fiscalDateEnding', ascending=False).reset_index(drop=True)

        meses = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
        df['periodo_fiscal'] = df['fiscalDateEnding'].dt.month.map(meses) + ' AF' + df['fiscalDateEnding'].dt.strftime('%y')

        df['Other'] = df['operatingIncome'] - df['incomeBeforeTax']
        df = df.drop(columns=['incomeBeforeTax'])

        cols_billion = ['totalRevenue','costOfRevenue','netIncome','operatingExpenses','operatingIncome',
                        'grossProfit','researchAndDevelopment','sellingGeneralAndAdministrative',
                        'incomeTaxExpense','Other']
        df = convert_columns_to_billions(df.copy(), cols_billion)
        round_numeric_df(df)

        df['netincome%'] = (df['netIncome'] / df['totalRevenue'] * 100).round(0)
        df['operatingIncome%'] = (df['operatingIncome'] / df['totalRevenue'] * 100).round(0)
        df['grossProfit%'] = (df['grossProfit'] / df['totalRevenue'] * 100).round(0)
        df['researchAndDevelopment%'] = (df['researchAndDevelopment'] / df['totalRevenue'] * 100).round(0)
        df['sellingGeneralAndAdministrative%'] = (df['sellingGeneralAndAdministrative'] / df['totalRevenue'] * 100).round(0)

        df = df.sort_values('fiscalDateEnding', ascending=False).reset_index(drop=True)
        df['previous_year_date'] = df['fiscalDateEnding'] - pd.DateOffset(years=1)
        previous_revenue = df.set_index('fiscalDateEnding')['totalRevenue']
        df['totalRevenue_previous_year'] = df['previous_year_date'].map(previous_revenue)
        df['totalRevenue_YoY'] = ((df['totalRevenue'] - df['totalRevenue_previous_year']) / df['totalRevenue_previous_year'] * 100).round(1)
        df = df.drop(columns=['totalRevenue_previous_year','previous_year_date'])

        rename_map = {
            'fiscalDateEnding': ticker, 'totalRevenue':'Ventas', 'costOfRevenue':'Costo de Ventas',
            'netIncome':'Utilidad neta', 'operatingExpenses':'Gastos operativos', 'operatingIncome':'EBIT',
            'grossProfit':'Utilidad Bruta', 'researchAndDevelopment':'Gasto en investigación y desarrollo',
            'sellingGeneralAndAdministrative':'Gastos de venta y administración',
            'incomeTaxExpense':'Gasto en impuestos', 'periodo_fiscal':'periodo_fiscal', 'Other':'Otros',
            'netincome%':'Utilidad neta %', 'operatingIncome%':'EBIT %', 'grossProfit%':'Utilidad Bruta %',
            'researchAndDevelopment%':'Gasto en investigación y desarrollo %',
            'sellingGeneralAndAdministrative%':'Gastos de venta y administración %',
            'totalRevenue_YoY':'Crecimiento Ventas Anual %'
        }
        return df.rename(columns=rename_map)


    def sankey_income_statement(df, periodo):
        fila = df[df["periodo_fiscal"] == periodo].iloc[0]

        ventas = fila["Ventas"]; costo_ventas = fila["Costo de Ventas"]
        utilidad_bruta = fila["Utilidad Bruta"]; gastos_operativos = fila["Gastos operativos"]
        ebit = fila["EBIT"]; rd = fila["Gasto en investigación y desarrollo"]
        sm = fila["Gastos de venta y administración"]; impuestos = fila["Gasto en impuestos"]
        utilidad_neta = fila["Utilidad neta"]

        otros_opex = max(gastos_operativos - rd - sm, 0)
        otros_ebit = max(ebit - impuestos - utilidad_neta, 0)

        pct_utilidad_bruta = utilidad_bruta / ventas * 100
        pct_ebit = ebit / ventas * 100
        pct_rd = rd / ventas * 100
        pct_sm = sm / ventas * 100
        pct_utilidad_neta = utilidad_neta / ventas * 100

        pct_ventas = None
        for col in df.columns:
            if col.strip().lower().startswith("crecimiento ventas"):
                pct_ventas = fila[col]
                break

        nombres = ["Ventas","Costo de Ventas","Utilidad Bruta","Gastos op.","EBIT","I+D",
                "Venta y Admin.","Otros gastos op.","Impuestos","Utilidad neta","Otros"]
        valores_nodos = [ventas, costo_ventas, utilidad_bruta, gastos_operativos, ebit, rd, sm,
                        otros_opex, impuestos, utilidad_neta, otros_ebit]
        pcts = [pct_ventas, None, pct_utilidad_bruta, None, pct_ebit, pct_rd, pct_sm, None, None,
                pct_utilidad_neta, None]

        def fmt_label(nombre, valor, pct):
            base = (f'<span style="font-size:12px; color:#000000;">{nombre}</span>'
                    f'<br><b style="font-size:13px; color:#1B2631;">${valor:,.1f}B</b>')
            if pct is not None:
                base += f'<br><span style="font-size:12px; color:#000000;">({pct:.0f}% Y/Y)</span>' if nombre == "Ventas" \
                        else f'<br><span style="font-size:12px; color:#000000;">({pct:.0f}%)</span>'
            return base

        labels = [fmt_label(n, v, p) for n, v, p in zip(nombres, valores_nodos, pcts)]

        node_colors = ["#B8A9F7","#E88AB8","#7C6CF5","#F2B8D4","#7C6CF5","#EC6FA8","#EC6FA8","#EC6FA8",
                        "#C2185B","#7C6CF5","#BDC3C7"]
        node_x = [0.001,0.32,0.32,0.62,0.62,0.90,0.90,0.90,0.90,0.90,0.90]
        node_y = [0.5,0.66,0.34,0.66,0.34,0.58,0.70,0.82,0.14,0.26,0.40]

        source = [0,0,2,2,3,3,3,4,4,4]
        target = [1,2,3,4,5,6,7,8,9,10]
        value = [costo_ventas, utilidad_bruta, gastos_operativos, ebit, rd, sm, otros_opex,
                impuestos, utilidad_neta, otros_ebit]
        link_colors = ["rgba(232,138,184,0.35)","rgba(184,169,247,0.35)","rgba(242,184,212,0.35)",
                        "rgba(184,169,247,0.35)","rgba(236,111,168,0.35)","rgba(236,111,168,0.35)",
                        "rgba(236,111,168,0.35)","rgba(194,24,91,0.35)","rgba(184,169,247,0.35)",
                        "rgba(189,195,199,0.35)"]

        fig = go.Figure(data=[go.Sankey(
            arrangement="fixed",
            node=dict(pad=22, thickness=16, line=dict(color="black", width=0.05),
                    label=labels, color=node_colors, x=node_x, y=node_y, hoverinfo="skip"),
            link=dict(source=source, target=target, value=value, color=link_colors, hoverinfo="skip"),
        )])

        fig.update_layout(
            title=dict(text=f"Estado de Resultados — {periodo}",
                        font=dict(family="Poppins, Helvetica Neue, Arial, sans-serif", size=16, color="#1B2631"), y=0.95),
            font=dict(size=14, color="#000000", family="Arial, sans-serif"),
            plot_bgcolor="white", paper_bgcolor="white",
            autosize=True, height=430, margin=dict(l=30, r=30, t=20, b=15),
        )
        return fig

    income_df_trim_1 = load_income_sankey_data(ticker)

    periodos_disponibles = income_df_trim_1["periodo_fiscal"].tolist()

    periodo_seleccionado = st.selectbox(
        "Periodo fiscal",
        options=periodos_disponibles,
        index=0,
        label_visibility="collapsed",
        key="periodo_sankey_dashboard"
    )

    fig = sankey_income_statement(
        income_df_trim_1,
        periodo=periodo_seleccionado
    )

    fig_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True},
        default_width="100%",
        default_height="430px"
    )

    box_html = f"""
    <div style="
        background: linear-gradient(180deg, #FAFBFC 0%, #F0F2F5 100%);
        border: 1px solid #E1E4E8;
        border-radius: 20px;
        padding: 1.25rem 1.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        box-sizing: border-box;
    ">
        {fig_html}
    </div>
    """

    components.html(
        box_html,
        height=520,
        scrolling=False
    )


    metricas_graficos = [
        #Crecimiento
        ("Crecimiento de ventas %", "%", "", 1),
        ("Crecimiento EBITDA %", "%", "", 1),
        ("Crecimiento EBIT %", "%", "", 1),
        ("Crecimiento de utilidad neta %", "%", "", 1),
        ("Crecimiento EPS %", "%", "", 1),
        #Estado de resultados
        ("Ventas", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Costo de ventas", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Utilidad bruta", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("EBITDA", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Utilidad operativa (EBIT)", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Utilidad neta", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Tasa impositiva %", "%", "", 1),
        #Reinversion y capex
        ("Tasa de reinversión %", "%", "", 1),
        ("Gastos de capital (CapEx)", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("CapEx de crecimiento", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("CapEx de mantenimiento", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Costo de mantenimiento %", "%", "", 1),
        ("Depreciación y amortización", "", "$", 2), # <-- cada 2 etiquetas y en $
        #Capital de trabajo y eficiencia operativa
        ("Capital de trabajo", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Cambio en capital de trabajo", "", "$", 2),
        ("Conversión de efectivo %", "%", "", 1),
        ("Costo de capital de trabajo %", "%", "", 1),
        ("Inventarios", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Cuentas por cobrar", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Cuentas por pagar", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Ingresos diferidos", "", "$", 2), # <-- cada 2 etiquetas y en $
        #Flujo de caja
        ("Flujo de caja libre", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Flujo de caja libre por acción", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Crecimiento del flujo de caja libre %", "%", "", 1),
        ("Crecimiento del FCF por acción %", "%", "", 1),
        ("Margen de flujo de caja libre %", "%", "", 1),
        #Estructura financiera y balance
        ("Efectivo e inversiones de corto plazo", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Inversiones de corto plazo", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Deuda a corto plazo", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Deuda a largo plazo", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Capital invertido", "", "$", 2), # <-- cada 2 etiquetas y en $
        ("Capital contable", "", "$", 2), # <-- cada 2 etiquetas y en $
        # otras métricas
        ("Utilidad por acción (EPS)", "", "$", 1),
        ("Crecimiento de acciones en circulación %", "%", "", 1),
        ("Recompras de acciones %", "%", "", 1),
        ("Pago de dividendos %", "%", "", 1),
        ("Crecimiento orgánico %", "%", "", 1),
        ("Puntuación total %", "%", "", 1),
    ]


    for metrica, sufijo, prefijo, label_step in metricas_graficos:
        if metrica not in tabla_vertical.index:
            st.warning(f"No se encontró la métrica '{metrica}' en la tabla.")
            continue

        serie = tabla_vertical.loc[metrica].astype(str).str.replace(',', '', regex=False)
        serie_num = pd.to_numeric(serie, errors='coerce')
        anios = [str(a) for a in tabla_vertical.loc[metrica].index]

        render_bar_card_html(
            labels=anios,
            values=serie_num.values,
            titulo=metrica.replace(" %", ""),
            sufijo=sufijo,
            prefijo=prefijo,
            label_step=label_step,
        )


elif current == "noticias":
    render_noticias_ui()


elif current == "glosario":

    st.markdown(_tv_css(), unsafe_allow_html=True)


    ESTADOS_FINANCIEROS = [
        {
            "nombre": "Balance General (BS)",
            "color": "#3B82F6",
            "pregunta": "¿Qué tienes y a quién le debes?",
            "tipo": "FOTO en un momento del tiempo",
            "formula": "Activo = Pasivo + Capital",
            "observa": ["Apalancamiento", "Liquidez", "Calidad de activos"],
        },
        {
            "nombre": "Estado de Resultados (IS)",
            "color": "#10B981",
            "pregunta": "¿Cuánto ganaste?",
            "tipo": "PELÍCULA de resultados",
            "formula": "Ingresos − Costos − Gastos = Utilidad",
            "observa": ["Crecimiento de ingresos YoY", "Calidad de márgenes"],
        },
        {
            "nombre": "Flujo de Efectivo (CF)",
            "color": "#F97316",
            "pregunta": "¿De dónde vino y a dónde se fue el efectivo?",
            "tipo": "PELÍCULA de caja",
            "formula": "Operación + Inversión + Financiamiento",
            "observa": [
                "¿La utilidad se vuelve caja?",
                "CapEx vs Depreciación",
                "¿Cómo gastan efectivo? (Dividendos, recompra, I+D)",
            ],
        },
        {
            "nombre": "Cambios en Capital",
            "color": "#EC4899",
            "pregunta": "¿Cómo cambió la inversión de los dueños?",
            "tipo": "PELÍCULA del equity",
            "formula": "Capital + Utilidad − Dividendos +/− Otros",
            "observa": [
                "Dilución",
                "Política de dividendos",
                "Recompras agresivas vs deuda creciente",
            ],
        },
    ]


    html2 = ['<div class="tv-card" style="margin-top:18px;">']
    html2.append('<div class="tv-title">Estados Financieros</div>')

    # Leyenda
    html2.append('<div class="tv-legend">')
    for ef in ESTADOS_FINANCIEROS:
        color = ef["color"]
        html2.append(
            f'<span class="tv-chip" style="background:{color}22;color:{color};">'
            f'<span class="tv-chip-dot" style="background:{color};"></span>{ef["nombre"]}</span>'
        )
    html2.append("</div>")

    # Grid de tarjetas (una por estado financiero)
    html2.append(
        '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:16px;">'
    )

    for ef in ESTADOS_FINANCIEROS:
        color = ef["color"]
        observa_items = "".join(
            f'<li style="margin-bottom:4px;">{item}</li>' for item in ef["observa"]
        )
        card_html = (
            f'<div style="background:#FAFAFC; border:1px solid #ECECF2; '
            f'border-left:4px solid {color}; border-radius:14px; padding:16px 18px;">'
            f'<div style="font-size:15px; font-weight:800; color:#1F1F1F; margin-bottom:6px;">'
            f'{ef["nombre"]}</div>'
            f'<div style="font-size:13px; color:#4A4A4A; margin-bottom:2px;">'
            f'{ef["pregunta"]}</div>'
            f'<div style="font-size:12px; font-weight:700; color:{color}; text-transform:uppercase; '
            f'letter-spacing:0.4px; margin-bottom:10px;">{ef["tipo"]}</div>'
            f'<div style="background:{color}15; color:{color}; font-family:Consolas, monospace; '
            f'font-weight:700; font-size:13px; padding:8px 12px; border-radius:8px; margin-bottom:12px;">'
            f'{ef["formula"]}</div>'
            f'<div style="font-size:11px; font-weight:700; color:#9A9AA5; text-transform:uppercase; '
            f'letter-spacing:0.5px; margin-bottom:6px;">¿Qué se observa principalmente?</div>'
            f'<ul style="font-size:13px; color:#333; margin:0; padding-left:18px;">{observa_items}</ul>'
            f'</div>'
        )
        html2.append(card_html)

    html2.append("</div>")

    # Banner final con la identidad contable
    banner_html = (
        f'<div style="margin-top:20px; background: linear-gradient(160deg, {PURPLE_DARK} 0%, '
        f'{PURPLE_MID} 55%, {PURPLE_DEEP} 100%); border-radius:16px; padding:20px 24px; text-align:center;">'
        f'<div style="color:{WHITE}; font-size:18px; font-weight:900; letter-spacing:0.5px; margin-bottom:6px;">'
        f'ACTIVO = PASIVO + CAPITAL</div>'
        f'<div style="color:{WHITE_MUTED}; font-size:13px; font-weight:500;">'
        f'Todo lo que la empresa TIENE = todo lo que la empresa DEBE + lo que aportaron los dueños</div>'
        f'</div>'
    )
    html2.append(banner_html)

    html2.append("</div>")  # cierra tv-card

    st.markdown("".join(html2), unsafe_allow_html=True)


    # ------------------------------------------------------------------
    # DATOS: Notas importantes — señales de una muy buena compañía
    # ------------------------------------------------------------------
    NOTAS_IMPORTANTES = [
        "Un ROIC alto a través del tiempo, a partir del 20% ya es alto y muy bueno, y debe ser constante en el tiempo.",
        "Capital invertido alto a través del tiempo.",
        "La combinación debe ser alta: ROIC alto y tasa de reinversión alta y en crecimiento.",
        "Flujo de caja y ROIC deben estar al alza.",
        "Ventas sólidas y al alza.",
        "Cuentas por cobrar al alza: bueno porque va a recibir dinero.",
        "El múltiplo preferido es EV/FCF.",
    ]

    # ------------------------------------------------------------------
    # RENDER: cuarta tarjeta — Notas importantes
    # ------------------------------------------------------------------
    html4 = ['<div class="tv-card" style="margin-top:18px;">']
    html4.append('<div class="tv-title">Notas importantes</div>')
    html4.append(
        '<div style="font-size:13px; font-weight:700; color:#7C6CF5; text-transform:uppercase; '
        'letter-spacing:0.5px; margin-bottom:14px;">Muy buena compañía para invertir</div>'
    )

    html4.append('<ol style="margin:0; padding-left:20px;">')
    for nota in NOTAS_IMPORTANTES:
        html4.append(
            '<li style="font-size:14px; color:#333; line-height:1.6; margin-bottom:10px;">'
            f'{nota}</li>'
        )
    html4.append("</ol>")

    html4.append("</div>")
    st.markdown("".join(html4), unsafe_allow_html=True)


    ratios_formulario = [

        # LIQUIDEZ
        {'Categoría': 'LIQUIDEZ', 'Ratio': 'Razón circulante',
         'Definición': 'Mide la capacidad de pagar deudas de corto plazo con activos circulantes. >1 = puedes cubrir.',
         'Fórmula': 'Activo Circulante / Pasivo Circulante'},
        {'Categoría': 'LIQUIDEZ', 'Ratio': 'Prueba ácida',
         'Definición': 'Mide la capacidad de pago de corto plazo excluyendo inventarios, por ser menos líquidos.',
         'Fórmula': '(Activo Circulante − Inventarios) / Pasivo Circulante'},
        {'Categoría': 'LIQUIDEZ', 'Ratio': 'Cash ratio',
         'Definición': 'Mide la capacidad de cubrir deudas de corto plazo únicamente con efectivo.',
         'Fórmula': 'Efectivo / Pasivo Circulante'},
        {'Categoría': 'LIQUIDEZ', 'Ratio': 'Capital trabajo neto',
         'Definición': 'Mide el colchón financiero disponible después de cubrir los pasivos de corto plazo.',
         'Fórmula': 'Activo Circulante − Pasivo Circulante'},

        # SOLVENCIA / APALANCAMIENTO
        {'Categoría': 'SOLVENCIA / APALANCAMIENTO', 'Ratio': 'Endeudamiento',
         'Definición': 'Mide qué porcentaje de los activos está financiado con deuda.',
         'Fórmula': 'Pasivo Total / Activo Total'},
        {'Categoría': 'SOLVENCIA / APALANCAMIENTO', 'Ratio': 'Deuda / Capital',
         'Definición': 'Mide cuánta deuda existe por cada peso de capital propio.',
         'Fórmula': 'Pasivo Total / Capital Contable'},
        {'Categoría': 'SOLVENCIA / APALANCAMIENTO', 'Ratio': 'Cobertura de intereses',
         'Definición': 'Mide cuántas veces la utilidad operativa puede cubrir los gastos financieros.',
         'Fórmula': 'EBIT / |Gastos Financieros|'},

        # RENTABILIDAD
        {'Categoría': 'RENTABILIDAD', 'Ratio': 'Margen bruto',
         'Definición': 'Mide qué porcentaje de las ventas queda después del costo directo de producción o compra.',
         'Fórmula': 'Utilidad Bruta / Ventas'},
        {'Categoría': 'RENTABILIDAD', 'Ratio': 'Margen operativo',
         'Definición': 'Mide el porcentaje de ventas que queda después de los gastos operativos.',
         'Fórmula': 'EBIT / Ventas'},
        {'Categoría': 'RENTABILIDAD', 'Ratio': 'Margen EBITDA',
         'Definición': 'Mide la generación operativa antes de depreciación, amortización, intereses e impuestos.',
         'Fórmula': 'EBITDA / Ventas'},
        {'Categoría': 'RENTABILIDAD', 'Ratio': 'Margen neto',
         'Definición': 'Mide cuánto de cada peso vendido se convierte finalmente en utilidad neta.',
         'Fórmula': 'Utilidad Neta / Ventas'},
        {'Categoría': 'RENTABILIDAD', 'Ratio': 'ROA',
         'Definición': 'Mide el rendimiento generado por los activos totales.',
         'Fórmula': 'Utilidad Neta / Activo Total'},
        {'Categoría': 'RENTABILIDAD', 'Ratio': 'ROE',
         'Definición': 'Mide el rendimiento generado sobre el capital de los accionistas.',
         'Fórmula': 'Utilidad Neta / Capital Contable'},

        # EFICIENCIA OPERATIVA
        {'Categoría': 'EFICIENCIA OPERATIVA', 'Ratio': 'Rotación de activos',
         'Definición': 'Mide cuántas veces los activos se convierten en ventas durante el periodo.',
         'Fórmula': 'Ventas / Activo Total'},
        {'Categoría': 'EFICIENCIA OPERATIVA', 'Ratio': 'Días de inventario (DIO)',
         'Definición': 'Mide cuántos días permanece el inventario antes de venderse.',
         'Fórmula': '(Inventarios / |COGS|) × 365'},
        {'Categoría': 'EFICIENCIA OPERATIVA', 'Ratio': 'Días de cobro (DSO)',
         'Definición': 'Mide el número promedio de días que tarda la empresa en cobrar a sus clientes.',
         'Fórmula': '(Cuentas por Cobrar / Ventas) × 365'},
        {'Categoría': 'EFICIENCIA OPERATIVA', 'Ratio': 'Días de pago (DPO)',
         'Definición': 'Mide el número promedio de días que tarda la empresa en pagar a sus proveedores.',
         'Fórmula': '(Cuentas por Pagar / |COGS|) × 365'},
        {'Categoría': 'EFICIENCIA OPERATIVA', 'Ratio': 'Ciclo conversión efectivo',
         'Definición': 'Mide los días entre el pago a proveedores y el cobro de las ventas.',
         'Fórmula': 'DIO + DSO − DPO'},

        # DUPONT
        {'Categoría': 'DUPONT', 'Ratio': 'Margen neto',
         'Definición': 'Mide cuánto gana la empresa por cada peso de ventas.',
         'Fórmula': 'Utilidad Neta / Ventas'},
        {'Categoría': 'DUPONT', 'Ratio': 'Rotación de activos',
         'Definición': 'Mide cuántas veces los activos generan ventas.',
         'Fórmula': 'Ventas / Activo Total'},
        {'Categoría': 'DUPONT', 'Ratio': 'Apalancamiento',
         'Definición': 'Mide cuánto se multiplican los recursos propios mediante el uso de activos financiados.',
         'Fórmula': 'Activo Total / Capital Contable'},
        {'Categoría': 'DUPONT', 'Ratio': 'ROE (DuPont)',
         'Definición': 'Descompone el ROE en margen neto, rotación de activos y apalancamiento.',
         'Fórmula': 'Margen × Rotación × Apalancamiento'},

        # CASH FLOW Y CALIDAD
        {'Categoría': 'CASH FLOW Y CALIDAD', 'Ratio': 'FCF',
         'Definición': 'Mide el flujo de efectivo disponible después de las operaciones y las inversiones.',
         'Fórmula': 'OCF + Flujo de Inversión'},
        {'Categoría': 'CASH FLOW Y CALIDAD', 'Ratio': 'Calidad de utilidades',
         'Definición': 'Mide qué tan bien la utilidad contable se convierte en flujo de efectivo.',
         'Fórmula': 'Flujo de Operación / Utilidad Neta'},
        {'Categoría': 'CASH FLOW Y CALIDAD', 'Ratio': 'CapEx / D&A',
         'Definición': 'Compara la inversión en activos con el gasto por depreciación y amortización.',
         'Fórmula': '|Flujo de Inversión| / D&A'},
    ]

    ratios_df = pd.DataFrame(ratios_formulario)

    # ------------------------------------------------------------------
    # COLORES POR CATEGORÍA (mismo patrón que CATEGORY_COLORS)
    # ------------------------------------------------------------------
    GLOSARIO_CATEGORY_COLORS = {
        "LIQUIDEZ": "#3B82F6",
        "SOLVENCIA / APALANCAMIENTO": "#F97316",
        "RENTABILIDAD": "#10B981",
        "EFICIENCIA OPERATIVA": "#8B5CF6",
        "DUPONT": "#EC4899",
        "CASH FLOW Y CALIDAD": "#EAB308",
    }

    # ------------------------------------------------------------------
    # RENDER: mismo estilo visual que "Finanzas clave" (tv-card, tv-legend, tv-table)
    # ------------------------------------------------------------------
    html = ['<div class="tv-card">']
    html.append('<div class="tv-title">Glosario de ratios financieros</div>')

    # Leyenda de categorías
    html.append('<div class="tv-legend">')
    for cat, color in GLOSARIO_CATEGORY_COLORS.items():
        html.append(
            f'<span class="tv-chip" style="background:{color}22;color:{color};">'
            f'<span class="tv-chip-dot" style="background:{color};"></span>{cat}</span>'
        )
    html.append("</div>")

    # Tabla
    html.append('<div class="tv-scroll">')
    html.append(
        '<table class="tv-table" style="min-width:900px;"><thead><tr>'
        '<th style="text-align:left;">Ratio</th>'
        '<th style="text-align:left;">Definición</th>'
        '<th style="text-align:left;">Fórmula</th>'
        '</tr></thead><tbody>'
    )

    for cat in ratios_df['Categoría'].unique():
        color = GLOSARIO_CATEGORY_COLORS.get(cat, "#9CA3AF")
        sub_df = ratios_df[ratios_df['Categoría'] == cat]

        html.append(
            f'<tr><td colspan="3" style="background:{color}15; font-weight:800; '
            f'color:{color}; padding:10px 14px;">{cat}</td></tr>'
        )

        for _, row in sub_df.iterrows():
            html.append(
                '<tr>'
                f'<td class="tv-metric-cell" style="position:static; white-space:normal;">'
                f'<span class="tv-dot" style="background:{color};"></span>{row["Ratio"]}</td>'
                f'<td style="text-align:left; white-space:normal; max-width:340px; color:#4A4A4A;">{row["Definición"]}</td>'
                f'<td style="text-align:left; white-space:normal; font-family:Consolas, monospace; '
                f'color:#4B3AD6; font-weight:600;">{row["Fórmula"]}</td>'
                '</tr>'
            )

    html.append("</tbody></table></div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


    # ------------------------------------------------------------------
    # DATOS: Glosario extendido de métricas (50 métricas, ENG/ESP)
    # ------------------------------------------------------------------
    tabla_metricas = pd.DataFrame([
        {"ID": 1, "Nombre (ENG)": "EBITDA Margin", "Nombre (ESP)": "Margen EBITDA",
         "Fórmula": "EBITDA / Ingresos totales × 100", "Estados financieros": "Estado de resultados",
         "Descripción": "Mide qué porcentaje de los ingresos queda como EBITDA antes de intereses, impuestos, depreciación y amortización. En general, un margen más alto y estable es favorable."},
        {"ID": 2, "Nombre (ENG)": "EBIT Margin", "Nombre (ESP)": "Margen EBIT",
         "Fórmula": "EBIT / Ingresos totales × 100", "Estados financieros": "Estado de resultados",
         "Descripción": "Mide qué porcentaje de los ingresos se convierte en utilidad operativa antes de intereses e impuestos. Un margen más alto y estable generalmente indica mayor eficiencia operativa."},
        {"ID": 3, "Nombre (ENG)": "Net income Margin", "Nombre (ESP)": "Margen de utilidad neta",
         "Fórmula": "Utilidad neta / Ingresos totales × 100", "Estados financieros": "Estado de resultados",
         "Descripción": "Mide la rentabilidad neta de la empresa respecto a sus ingresos. Indica qué % de los ingresos se convierte en utilidad neta después de considerar todos los costos, gastos, intereses e impuestos."},
        {"ID": 4, "Nombre (ENG)": "Net Interest", "Nombre (ESP)": "Interés neto",
         "Fórmula": "Ingresos por intereses − Gastos por intereses", "Estados financieros": "Estado de resultados",
         "Descripción": "Mide el efecto neto de los ingresos y gastos por intereses. Un valor positivo significa que los ingresos por intereses superan los gastos por intereses."},
        {"ID": 5, "Nombre (ENG)": "Tax Rate", "Nombre (ESP)": "Tasa efectiva de impuestos",
         "Fórmula": "Gasto por impuestos / Utilidad antes de impuestos × 100", "Estados financieros": "Estado de resultados",
         "Descripción": "Mide la proporción de la utilidad antes de impuestos que se destina al pago de impuestos."},
        {"ID": 6, "Nombre (ENG)": "Maintenance CapEx", "Nombre (ESP)": "CapEx de mantenimiento",
         "Fórmula": "Min(CapEx, D&A)", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Se debe ir manteniendo. Estima la inversión mínima necesaria para mantener la capacidad operativa de los activos existentes."},
        {"ID": 7, "Nombre (ENG)": "Growth CapEx", "Nombre (ESP)": "CapEx de crecimiento",
         "Fórmula": "Total CapEx − Maintenance CapEx", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Inversión discrecional y variable que la empresa realiza únicamente en los años que busca expandir su capacidad productiva o escala de negocio. Está asociada con expansión, crecimiento de capacidad, nuevos proyectos, nuevas instalaciones, etc."},
        {"ID": 8, "Nombre (ENG)": "Deferred Revenue", "Nombre (ESP)": "Ingresos diferidos",
         "Fórmula": "No aplica", "Estados financieros": "Balance general",
         "Descripción": "Representa pagos recibidos por bienes o servicios que la empresa aún no ha entregado y, por lo tanto, todavía no puede reconocer como ingreso."},
        {"ID": 9, "Nombre (ENG)": "Estimated EBITDA", "Nombre (ESP)": "EBITDA estimado",
         "Fórmula": "EBIT + Depreciación y amortización", "Estados financieros": "Flujo de efectivo, Estado de Resultados",
         "Descripción": "Mide el resultado operativo antes de intereses, impuestos, depreciación y amortización. Se estima sumando la depreciación y amortización al EBIT."},
        {"ID": 10, "Nombre (ENG)": "Working capital", "Nombre (ESP)": "Capital de Trabajo Operativo",
         "Fórmula": "Capital de trabajo operativo = Activos circulantes − Pasivos circulantes. Donde AC: Inventarios, Cuentas por cobrar y PC: Cuentas por pagar, Ingresos diferidos",
         "Estados financieros": "Balance general",
         "Descripción": "Mide los recursos netos necesarios para financiar las operaciones de la empresa. Debe ser bajo o negativo."},
        {"ID": 11, "Nombre (ENG)": "Change In WC", "Nombre (ESP)": "Cambio en el Capital de Trabajo",
         "Fórmula": "Variación entre periodos", "Estados financieros": "Balance general",
         "Descripción": "Cambio en capital de trabajo al alza: Es desfavorable, ya que implica una mayor cantidad de recursos inmovilizados en la operación y, por tanto, una menor generación de flujo de efectivo."},
        {"ID": 12, "Nombre (ENG)": "Free Cash Flow", "Nombre (ESP)": "Flujo de Efectivo Libre",
         "Fórmula": "EBITDA − Intereses − Impuestos − CapEx − ΔWC",
         "Estados financieros": "Flujo de efectivo, Estado de Resultados, Balance General",
         "Descripción": "El efectivo que queda después de cubrir los gastos financieros, impuestos, inversión necesaria para mantener la operación y las necesidades de capital de trabajo. Entre más me quede mejor. Tiene que ser positivo y alto."},
        {"ID": 13, "Nombre (ENG)": "Free Cash Flow per Share", "Nombre (ESP)": "Flujo de efectivo libre por acción",
         "Fórmula": "FCF / Acciones en circulación", "Estados financieros": "Flujo de efectivo, Balance general",
         "Descripción": "Entre más alto mejor. Indica cuánto flujo de efectivo libre corresponde a cada acción de la empresa."},
        {"ID": 14, "Nombre (ENG)": "Maintenance CapEx Margin", "Nombre (ESP)": "Margen de CapEx de mantenimiento",
         "Fórmula": "CapEx de mantenimiento / Ingresos totales × 100", "Estados financieros": "Flujo de efectivo, Estado de resultados",
         "Descripción": "Mide qué porcentaje de los ingresos se destina a mantener la capacidad operativa existente. Un porcentaje más alto implica una mayor necesidad de reinversión para mantener el negocio."},
        {"ID": 15, "Nombre (ENG)": "Working Capital Margin", "Nombre (ESP)": "Margen de capital de trabajo",
         "Fórmula": "Capital de trabajo operativo / Ingresos totales × 100", "Estados financieros": "Balance general, Estado de resultados",
         "Descripción": "Mide cuánto capital de trabajo operativo requiere la empresa en relación con sus ingresos. Un requerimiento menor puede ser favorable porque implica menor capital inmovilizado en la operación."},
        {"ID": 16, "Nombre (ENG)": "Free Cash Flow Margin", "Nombre (ESP)": "Margen de flujo de efectivo libre",
         "Fórmula": "Flujo de efectivo libre / Ingresos totales × 100", "Estados financieros": "Flujo de efectivo, Estado de Resultados",
         "Descripción": "Indica qué porcentaje de los ingresos de la empresa se convierte en flujo de efectivo libre. Entre más alto mejor."},
        {"ID": 17, "Nombre (ENG)": "Cash Conversion Ratio", "Nombre (ESP)": "Ratio de Conversión de Efectivo",
         "Fórmula": "Flujo de efectivo libre / EBITDA × 100", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Evalúa la capacidad de la empresa para transformar su rentabilidad operativa (EBITDA) en efectivo realmente disponible. Entre más alto mejor."},
        {"ID": 18, "Nombre (ENG)": "Invested Capital", "Nombre (ESP)": "Capital invertido",
         "Fórmula": "Patrimonio de los accionistas + Deuda de corto plazo + Deuda de largo plazo + Obligaciones por arrendamientos de capital − Inversiones de corto plazo",
         "Estados financieros": "Balance general",
         "Descripción": "Alto a través del tiempo. Representa el capital que la empresa tiene comprometido para financiar sus operaciones y generar beneficios."},
        {"ID": 19, "Nombre (ENG)": "Return on Assets (ROA)", "Nombre (ESP)": "Rendimiento sobre activos (ROA)",
         "Fórmula": "Utilidad neta / Activos totales × 100", "Estados financieros": "Estado de resultados, Balance general",
         "Descripción": "ROA por encima de 10 o 15 es alto. Mide qué tan eficientemente la empresa utiliza sus activos totales para generar utilidad neta."},
        {"ID": 20, "Nombre (ENG)": "Return on Equity (ROE)", "Nombre (ESP)": "Rendimiento sobre el capital (ROE)",
         "Fórmula": "(Utilidad neta / Capital contable) × 100", "Estados financieros": "Estado de resultados, Balance general",
         "Descripción": "Mide qué tan eficientemente una empresa genera utilidad. ROE se mira más que el ROA. ROE por encima de 10 o 15 es alto."},
        {"ID": 21, "Nombre (ENG)": "Net Operating Profit After Tax (NOPAT)", "Nombre (ESP)": "Utilidad operativa después de impuestos (NOPAT)",
         "Fórmula": "EBIT × (1 − Tasa efectiva de impuestos)", "Estados financieros": "Estado de resultados",
         "Descripción": "Mide la utilidad generada por las operaciones después de impuestos, sin considerar el efecto de la estructura de financiamiento."},
        {"ID": 22, "Nombre (ENG)": "Return on Invested Capital (ROIC)", "Nombre (ESP)": "Rendimiento sobre el Capital Invertido (ROIC)",
         "Fórmula": "(NOPAT / Capital invertido) × 100", "Estados financieros": "Estado de resultados, Balance general",
         "Descripción": "Mide qué tan eficientemente una empresa utiliza el capital invertido en su operación para generar utilidad operativa después de impuestos. Un ROIC a partir de 20% ya es alto y muy bueno, y que sea constante en el tiempo."},
        {"ID": 23, "Nombre (ENG)": "Reinvestment Rate", "Nombre (ESP)": "Tasa de reinversión",
         "Fórmula": "Growth CapEx / FCF × 100", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Mide qué proporción del FCF se destina, según tu definición, a inversiones orientadas al crecimiento. Una tasa alta indica que una mayor proporción del FCF se está reinvirtiendo."},
        {"ID": 24, "Nombre (ENG)": "Dividends Paid", "Nombre (ESP)": "Dividendos pagados",
         "Fórmula": "Dividendos de acciones comunes + Dividendos de acciones preferentes", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Representa el efectivo pagado a los accionistas en forma de dividendos durante el periodo."},
        {"ID": 25, "Nombre (ENG)": "Share Repurchases", "Nombre (ESP)": "Recompras de acciones",
         "Fórmula": "Recompras de acciones comunes + Recompras de acciones preferentes", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Representa el efectivo utilizado por la empresa para recomprar sus propias acciones."},
        {"ID": 26, "Nombre (ENG)": "Total Debt", "Nombre (ESP)": "Deuda total",
         "Fórmula": "Deuda de corto plazo + Deuda de largo plazo", "Estados financieros": "Balance general",
         "Descripción": "Representa las obligaciones financieras de corto y largo plazo de la empresa."},
        {"ID": 27, "Nombre (ENG)": "Debt Repayment", "Nombre (ESP)": "Pago de deuda",
         "Fórmula": "Reducción de deuda entre periodos", "Estados financieros": "Balance general",
         "Descripción": "Mide la reducción de la deuda total respecto al periodo anterior. Un valor positivo indica que la empresa redujo su deuda."},
        {"ID": 28, "Nombre (ENG)": "Growth CapEx / FCF", "Nombre (ESP)": "Reinversión en crecimiento / FCF",
         "Fórmula": "Growth CapEx / FCF × 100", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Mide qué proporción del FCF se destina a inversiones para crecimiento."},
        {"ID": 29, "Nombre (ENG)": "Free Cash Flow Payout Ratio", "Nombre (ESP)": "Ratio de distribución de dividendos",
         "Fórmula": "Dividendos pagados / Flujo de efectivo libre × 100", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Los dividendos se dan cuando la compañía ya no va a crecer tanto. Mide la capacidad del flujo de efectivo libre para cubrir los dividendos. Un porcentaje bajo indica que la empresa conserva una mayor proporción de su FCF después de pagar dividendos; un porcentaje alto indica que una mayor parte del FCF se destina a dividendos."},
        {"ID": 30, "Nombre (ENG)": "Buybacks %", "Nombre (ESP)": "Recompra de acciones %",
         "Fórmula": "(Recompras de Acciones / Flujo de Caja Libre) × 100", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Mide qué porcentaje del flujo de efectivo libre (FCF) se destina a la recompra de acciones. Un porcentaje alto indica que una mayor proporción del efectivo disponible para los accionistas se utiliza para recompras."},
        {"ID": 31, "Nombre (ENG)": "Debt Repayment / FCF", "Nombre (ESP)": "Amortización de deuda / FCF",
         "Fórmula": "Pago de deuda / FCF × 100", "Estados financieros": "Flujo de efectivo, Balance general",
         "Descripción": "Mide qué proporción del FCF se utiliza para reducir deuda. Un porcentaje alto indica una mayor asignación del FCF a desapalancamiento."},
        {"ID": 32, "Nombre (ENG)": "Total %", "Nombre (ESP)": "Puntuación total %",
         "Fórmula": "Crecimiento orgánico % + Dividendos % + Recompras % + Amortización de deuda %", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Representa qué proporción del flujo de efectivo se asigna a crecimiento, dividendos, recompras y reducción de deuda. No debe superar 100%. Si es menor a 100%, la diferencia puede permanecer como efectivo o destinarse a otros usos, como adquisiciones o inversiones."},
        {"ID": 33, "Nombre (ENG)": "Total Debt", "Nombre (ESP)": "Deuda total",
         "Fórmula": "Deuda corriente de largo plazo + Obligaciones por arrendamientos + Deuda de largo plazo", "Estados financieros": "Balance general",
         "Descripción": "Representa las obligaciones financieras de largo plazo y los arrendamientos considerados como deuda."},
        {"ID": 34, "Nombre (ENG)": "Debt-to-Equity Ratio (D/E)", "Nombre (ESP)": "Ratio Deuda/Capital",
         "Fórmula": "Deuda total / Patrimonio", "Estados financieros": "Balance general",
         "Descripción": "Mide el nivel de deuda de la empresa en relación con el capital aportado por los accionistas. Un valor más bajo generalmente implica menor apalancamiento financiero."},
        {"ID": 35, "Nombre (ENG)": "Current Ratio", "Nombre (ESP)": "Razón circulante",
         "Fórmula": "Activos circulantes / Pasivos circulantes", "Estados financieros": "Balance general",
         "Descripción": "Mide la capacidad de la empresa para cubrir sus obligaciones de corto plazo con sus activos circulantes. Un valor > 1 indica cobertura; > 1.5 puede ser una referencia favorable."},
        {"ID": 36, "Nombre (ENG)": "5-Year Revenue CAGR", "Nombre (ESP)": "Crecimiento de ingresos a 5 años",
         "Fórmula": "[(Ingresos actuales / Ingresos de hace 5 años)^(1/5) − 1] × 100", "Estados financieros": "Estado de resultados",
         "Descripción": "Mide la tasa de crecimiento anual compuesta de los ingresos durante los últimos 5 años."},
        {"ID": 37, "Nombre (ENG)": "Gross Profit Margin", "Nombre (ESP)": "Margen bruto",
         "Fórmula": "Utilidad bruta / Ingresos totales × 100", "Estados financieros": "Estado de resultados",
         "Descripción": "Mide qué porcentaje de los ingresos queda después de cubrir el costo de ventas. Un margen más alto y estable generalmente indica mayor capacidad para generar utilidad antes de gastos operativos."},
        {"ID": 38, "Nombre (ENG)": "Net Debt", "Nombre (ESP)": "Deuda neta",
         "Fórmula": "Deuda total − Efectivo e inversiones de corto plazo", "Estados financieros": "Balance general",
         "Descripción": "Mide la deuda financiera de la empresa después de descontar el efectivo y las inversiones de corto plazo disponibles para reducirla. Una deuda neta menor generalmente implica menor apalancamiento."},
        {"ID": 39, "Nombre (ENG)": "Net Debt / EBITDA", "Nombre (ESP)": "Deuda neta / EBITDA",
         "Fórmula": "Deuda neta / EBITDA", "Estados financieros": "Balance general, Estado de resultados",
         "Descripción": "Mide la deuda neta en relación con la capacidad operativa de generación de EBITDA. Un múltiplo más bajo generalmente indica menor apalancamiento. Ejemplo: 2.0x significa una deuda neta equivalente a 2 veces el EBITDA."},
        {"ID": 40, "Nombre (ENG)": "Market Capitalization", "Nombre (ESP)": "Capitalización bursátil",
         "Fórmula": "Acciones en circulación × Precio por acción", "Estados financieros": "Balance general, Datos de mercado",
         "Descripción": "Representa el valor de mercado del capital de los accionistas."},
        {"ID": 41, "Nombre (ENG)": "Enterprise Value (EV)", "Nombre (ESP)": "Valor empresa",
         "Fórmula": "Capitalización bursátil + Deuda neta", "Estados financieros": "Balance general, Datos de mercado",
         "Descripción": "Representa el valor de la operación de la empresa considerando el valor del capital y la deuda neta. Es utilizado en múltiplos como EV/EBITDA, EV/EBIT y EV/FCF."},
        {"ID": 42, "Nombre (ENG)": "Price-to-Earnings Ratio (P/E o PER)", "Nombre (ESP)": "Precio por acción sobre utilidad por acción",
         "Fórmula": "Capitalización bursátil / Utilidad neta", "Estados financieros": "Estado de resultados, Datos de mercado",
         "Descripción": "Mide cuánto está pagando el mercado por cada unidad monetaria de utilidad neta. Un PER de 34.8x significa que la capitalización bursátil equivale a 34.8 veces la utilidad neta anual."},
        {"ID": 43, "Nombre (ENG)": "Enterprise Value / Free Cash Flow (EV/FCF)", "Nombre (ESP)": "Múltiplo Valor Empresa / Flujo de Efectivo Libre",
         "Fórmula": "Valor Empresa / FCF", "Estados financieros": "Flujo de efectivo, Balance general",
         "Descripción": "Es el mejor múltiplo. Mide cuántas veces el Flujo de Efectivo Libre anual está pagando el mercado por el valor total de la empresa."},
        {"ID": 44, "Nombre (ENG)": "Enterprise Value / EBITDA (EV/EBITDA)", "Nombre (ESP)": "Múltiplo Valor Empresa / EBITDA",
         "Fórmula": "Valor Empresa / EBITDA", "Estados financieros": "Flujo de efectivo, Balance general",
         "Descripción": "Múltiplo de valuación utilizado para comparar cuánto está pagando el mercado por la capacidad operativa de generación de beneficios de una empresa. Ejemplo: EV/EBITDA = 8.0x. El Valor Empresa de la compañía equivale a 8 veces su EBITDA."},
        {"ID": 45, "Nombre (ENG)": "Enterprise Value / EBIT (EV/EBIT)", "Nombre (ESP)": "Múltiplo Valor Empresa / EBIT",
         "Fórmula": "Valor Empresa / EBIT", "Estados financieros": "Estado de resultados, Balance general, Datos de mercado",
         "Descripción": "Mide cuántas veces el EBIT está valorando el mercado respecto al Valor Empresa. Un múltiplo menor puede indicar una valuación más atractiva, siempre que se compare con empresas similares y considerando crecimiento y riesgo."},
        {"ID": 46, "Nombre (ENG)": "Profitability", "Nombre (ESP)": "Rendimiento anualizado implícito",
         "Fórmula": "No aplica", "Estados financieros": "No aplica",
         "Descripción": "Rentabilidad anualizada esperada al comparar el valor intrínseco estimado a 5 años con el precio actual. Ejemplo 1.47 → 1.47% de rentabilidad anual y CETES está en 7%, es mejor CETES."},
        {"ID": 47, "Nombre (ENG)": "Cash and Cash Equivalents at Carrying Value", "Nombre (ESP)": "Efectivo y equivalentes de efectivo",
         "Fórmula": "No aplica", "Estados financieros": "Balance general",
         "Descripción": "Representa la liquidez inmediata que tiene la empresa para hacer frente a obligaciones y aprovechar oportunidades de inversión (ejemplo: compra de nuevas compañías o algo). Efectivo: Objetivo ideal en 0; evitar capital estancado o improductivo."},
        {"ID": 48, "Nombre (ENG)": "Cash and Short-Term Investments", "Nombre (ESP)": "Efectivo e inversiones de corto plazo",
         "Fórmula": "No aplica", "Estados financieros": "Balance general",
         "Descripción": "Dinero disponible de la empresa más inversiones financieras de corto plazo y alta liquidez que pueden convertirse rápidamente en efectivo."},
        {"ID": 49, "Nombre (ENG)": "Stock-Based Compensation (SBC)", "Nombre (ESP)": "Compensación basada en acciones",
         "Fórmula": "No aplica", "Estados financieros": "Flujo de efectivo",
         "Descripción": "Acciones del empleado."},
        {"ID": 50, "Nombre (ENG)": "Common Stock Shares Outstanding", "Nombre (ESP)": "Acciones en circulación",
         "Fórmula": "No aplica", "Estados financieros": "Balance general",
         "Descripción": "Tiene que ser bajo y estable. Representa el número de acciones ordinarias en circulación. Más que buscar un nivel absoluto bajo, es importante analizar su evolución: un aumento persistente puede generar dilución, mientras que una reducción puede incrementar el beneficio y el flujo de efectivo por acción."},
    ])

    # ------------------------------------------------------------------
    # COLORES POR ESTADO FINANCIERO PRINCIPAL
    # ------------------------------------------------------------------
    METRICAS_CATEGORY_COLORS = {
        "Estado de resultados": "#10B981",
        "Balance general": "#3B82F6",
        "Flujo de efectivo": "#F97316",
        "Datos de mercado": "#EC4899",
        "No aplica": "#9CA3AF",
    }

    # Estado financiero "principal" = el primero listado en la columna
    tabla_metricas["_categoria_principal"] = (
        tabla_metricas["Estados financieros"].str.split(",").str[0].str.strip()
    )

    # ------------------------------------------------------------------
    # RENDER: tercera tarjeta — Glosario extendido de métricas (50)
    # ------------------------------------------------------------------
    html3 = ['<div class="tv-card" style="margin-top:18px;">']
    html3.append('<div class="tv-title">Glosario extendido de métricas (ENG / ESP)</div>')

    # Leyenda de categorías
    html3.append('<div class="tv-legend">')
    for cat, color in METRICAS_CATEGORY_COLORS.items():
        html3.append(
            f'<span class="tv-chip" style="background:{color}22;color:{color};">'
            f'<span class="tv-chip-dot" style="background:{color};"></span>{cat}</span>'
        )
    html3.append("</div>")

    # Tabla
    html3.append('<div class="tv-scroll">')
    html3.append(
        '<table class="tv-table" style="min-width:1200px;"><thead><tr>'
        '<th style="text-align:left;">#</th>'
        '<th style="text-align:left;">Nombre (ENG)</th>'
        '<th style="text-align:left;">Nombre (ESP)</th>'
        '<th style="text-align:left;">Fórmula</th>'
        '<th style="text-align:left;">Estados financieros</th>'
        '<th style="text-align:left;">Descripción</th>'
        '</tr></thead><tbody>'
    )

    for _, row in tabla_metricas.iterrows():
        color = METRICAS_CATEGORY_COLORS.get(row["_categoria_principal"], "#9CA3AF")
        html3.append(
            '<tr>'
            f'<td style="text-align:left; color:#9A9AA5; font-weight:700;">{row["ID"]}</td>'
            f'<td class="tv-metric-cell" style="position:static; white-space:normal;">'
            f'<span class="tv-dot" style="background:{color};"></span>{row["Nombre (ENG)"]}</td>'
            f'<td style="text-align:left; white-space:normal; color:#1F1F1F; font-weight:600;">{row["Nombre (ESP)"]}</td>'
            f'<td style="text-align:left; white-space:normal; font-family:Consolas, monospace; '
            f'color:#4B3AD6; font-weight:600; max-width:260px;">{row["Fórmula"]}</td>'
            f'<td style="text-align:left; white-space:normal; color:{color}; font-weight:700; max-width:180px;">{row["Estados financieros"]}</td>'
            f'<td style="text-align:left; white-space:normal; color:#4A4A4A; max-width:420px;">{row["Descripción"]}</td>'
            '</tr>'
        )

    html3.append("</tbody></table></div></div>")
    st.markdown("".join(html3), unsafe_allow_html=True)