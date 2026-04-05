#Librerias obligatorias
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np

#modelo
from sklearn.ensemble import GradientBoostingClassifier


#graficos
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.figure_factory as ff
import time
from plotly.subplots import make_subplots


_DATA_DIR = Path(__file__).resolve().parent
panorama_df = pd.read_csv(_DATA_DIR / "panorama_df.csv")

# Configuración inicial
st.set_page_config(layout="wide")

# ==============================
# APP
# ==============================
st.title("📊 Macro Dashboard")

# ==============================
# KPIs
# ==============================
c1, c2, c3, c4 = st.columns(4)

c1.metric("WTI", round(panorama_df["crudeoil_wti_value"].iloc[-1], 2))
c2.metric("Bono 10Y", round(panorama_df["bono10_value"].iloc[-1], 2))
c3.metric("Bono 2Y", round(panorama_df["bono2_value"].iloc[-1], 2))
c4.metric("Yield Curve", round(panorama_df["yield_curve"].iloc[-1], 2))

st.markdown("---")

# ==============================
# FIG 3: YIELD CURVE
# ==============================
fig3 = go.Figure()

# Línea principal
fig3.add_trace(go.Scatter(
    x=panorama_df["date"],
    y=panorama_df["yield_curve"],
    name="Yield Curve",
    line=dict(color="blue", width=1.5),
    fill="tozeroy"
))

# Puntos de inversión
invertida = panorama_df[panorama_df["yield_curve_neg"] == 1]

fig3.add_trace(go.Scatter(
    x=invertida["date"],
    y=invertida["yield_curve"],
    mode="markers",
    name="Curva invertida",
    marker=dict(color="red", size=4)
))

# Línea cero
fig3.add_hline(
    y=0,
    line_dash="dash",
    line_color="red"
)

# ==============================
# SOMBREADO DE CRISIS
# ==============================
crisis = [
    ('2014-10-01', '2016-02-28', 'Caída petróleo'),
    ('2018-10-01', '2019-01-31', 'Fed'),
    ('2020-02-01', '2020-05-31', 'COVID'),
    ('2022-01-01', '2022-10-31', 'Inflación/tasas'),
]

for start, end, label in crisis:
    fig3.add_vrect(
        x0=start,
        x1=end,
        fillcolor="rgba(255, 80, 80, 0.12)",
        line_width=0,
        layer="below",  # 👈 importante para que no tape la gráfica
        annotation_text=label,
        annotation_position="top left",
        annotation_font_size=10
    )

# Layout
fig3.update_layout(
    title="Yield Curve (10Y - 2Y) con periodos de estrés",
    template="plotly_white",
    height=400,
    hovermode="x unified"
)

# ==============================
# FIG 4: BONOS
# ==============================
fig4 = make_subplots()

fig4.add_trace(go.Scatter(
    x=panorama_df["date"],
    y=panorama_df["bono10_value"],
    name="Bono 10Y"
))

fig4.add_trace(go.Scatter(
    x=panorama_df["date"],
    y=panorama_df["bono2_value"],
    name="Bono 2Y"
))


crisis = [
    ('2014-10-01', '2016-02-28', 'Caída petróleo'),
    ('2018-10-01', '2019-01-31', 'Fed'),
    ('2020-02-01', '2020-05-31', 'COVID'),
    ('2022-01-01', '2022-10-31', 'Inflación/tasas'),
]

for start, end, label in crisis:
    fig4.add_vrect(
        x0=start, x1=end,
        fillcolor="rgba(255,0,0,0.1)",
        line_width=0,
        annotation_text=label
    )

fig4.update_layout(
    title="Bonos 2Y vs 10Y",
    template="plotly_white",
    height=400
)


#variable objetivo ------------------------------------------------------------------------------------
panorama_df['date'] = pd.to_datetime(panorama_df['date'])
panorama_df = panorama_df.sort_values('date').reset_index(drop=True)
panorama_df = panorama_df.drop(columns=['Unnamed: 0'])


# Opcion 1: Mercado favorable

HORIZONTE = 5

# Retorno futuro del WTI
panorama_df['wti_ret_forward5'] = (
    panorama_df['crudeoil_wti_value'].shift(-HORIZONTE) /
    panorama_df['crudeoil_wti_value'] - 1
)

# Retornos recientes para detectar régimen actual
panorama_df['gold_ret_5d']   = panorama_df['gold_price'].pct_change(5)
panorama_df['bono10_ret_5d'] = panorama_df['bono10_value'].pct_change(5)
panorama_df['wti_ret_5d_now'] = panorama_df['crudeoil_wti_value'].pct_change(5)

#Definición de regímenes (umbrales que puedo modificar)
UMBRAL_CAIDA_FUERTE = -0.05   # WTI cae más del 5% en 5 días → riesgo
UMBRAL_SUBIDA       =  0.01   # Subida mínima para considerar tendencia alcista

# Régimen 1: CRECIMIENTO — petróleo sube + bonos a 10Y suben
panorama_df['regimen_crecimiento'] = (
    (panorama_df['wti_ret_5d_now'] > UMBRAL_SUBIDA) &
    (panorama_df['bono10_ret_5d']  > UMBRAL_SUBIDA)
).astype(int)

# Régimen 2: MIEDO/GEOPOLÍTICO — oro sube + petróleo sube
panorama_df['regimen_miedo'] = (
    (panorama_df['gold_ret_5d']    > UMBRAL_SUBIDA) &
    (panorama_df['wti_ret_5d_now'] > UMBRAL_SUBIDA)
).astype(int)

# Régimen 3: RIESGO — petróleo cae fuerte
panorama_df['regimen_riesgo'] = (
    panorama_df['wti_ret_5d_now'] < UMBRAL_CAIDA_FUERTE
).astype(int)

# Régimen 4: ESTANFLACIÓN — petróleo sube + bonos caen (presión inflacionaria)
panorama_df['regimen_estanflacion'] = (
    (panorama_df['wti_ret_5d_now'] > UMBRAL_SUBIDA) &
    (panorama_df['bono10_ret_5d']  < -UMBRAL_SUBIDA)
).astype(int)

# Variable objetivo
# FAVORABLE: WTI sube en los próximos 5 días + régimen actual es crecimiento (excluye miedo y riesgo)
panorama_df['mercado_favorable'] = (
    (panorama_df['wti_ret_forward5'] > 0) &          # WTI sube a futuro
    (panorama_df['yield_curve'] > 0) &               # Curva no invertida
    (panorama_df['regimen_riesgo'] == 0) &            # No hay caída fuerte actual
    (panorama_df['regimen_miedo'] == 0)               # No es rally por miedo
).astype(int)

# Opcion 2: Mercado favorable: regimen_actual (multiclase que son 4 opciones)

# 0 = Riesgo, 1 = Neutro, 2 = Crecimiento, 3 = Miedo

def asignar_regimen_multiclase(row):
    if row['regimen_riesgo'] == 1:
        return 0  # Riesgo (peor)
    elif row['regimen_crecimiento'] == 1 and row['regimen_miedo'] == 0:
        return 2  # Crecimiento limpio (mejor)
    elif row['regimen_miedo'] == 1:
        return 3  # Miedo/Geopolítico (volátil)
    else:
        return 1  # Neutro

panorama_df['regimen_actual'] = panorama_df.apply(asignar_regimen_multiclase, axis=1)

# Eliminar nulos
panorama_df = panorama_df.dropna()

# MODELO -----------------------------------------------------------
FEATURES = [
    "bono10_rsi14",
    "gold_ret_21d",
    "wti_ret_5d",
    "wti_ret_21d",
    "gold_52w_pct",
    "wti_rsi14",
    "wti_vol20_ann",
    "wti_52w_pct",
    "wti_drawdown",
    "gold_vol20_ann",
    "gold_drawdown",
    "wti_brent_spread",
    "bono10_vol20_ann",
    "gold_oil_ratio",
    "yield_curve_neg",
    "regimen_miedo",
    "bono2_drawdown",
    "gold_price"
]

X     = panorama_df[FEATURES]
y     = panorama_df['mercado_favorable']
dates = panorama_df['date']

# Split temporal 80/20
TRAIN_FRAC = 0.80
split_idx  = int(len(X) * TRAIN_FRAC)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test       = dates.iloc[split_idx:]

#entrenamiento
modelo = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42
)

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

#prediccion

#Prediccion (ingresando datos de forma manual)
wti    = 104.69
brent  = 121.88
bono2  = 3.82
bono10 = 4.35
gold   = 4425.95
#-------------------------------------------------------------------
from datetime import date

nueva_fila = {
    'date':                 pd.Timestamp(date.today()),
    'crudeoil_wti_value':   wti,
    'crudeoil_brent_value': brent,
    'bono2_value':          bono2,
    'bono10_value':         bono10,
    'gold_price':           gold,
}

for col in panorama_df.columns:
    if col not in nueva_fila:
        nueva_fila[col] = np.nan

df_extendido = pd.concat(
    [panorama_df, pd.DataFrame([nueva_fila])],
    ignore_index=True
)

# ── 2. FEATURES (idénticos al entrenamiento) ─────────────────────────────────
activos = {
    'wti':    'crudeoil_wti_value',
    'brent':  'crudeoil_brent_value',
    'bono2':  'bono2_value',
    'bono10': 'bono10_value',
    'gold':   'gold_price'
}

# 1. Retornos
for nombre, col in activos.items():
    df_extendido[f'{nombre}_ret_1d']  = df_extendido[col].pct_change(1)
    df_extendido[f'{nombre}_ret_5d']  = df_extendido[col].pct_change(5)
    df_extendido[f'{nombre}_ret_21d'] = df_extendido[col].pct_change(21)
    df_extendido[f'{nombre}_logret']  = np.log(df_extendido[col] / df_extendido[col].shift(1))
    df_extendido[f'{nombre}_ma20']    = df_extendido[col].rolling(20).mean()
    df_extendido[f'{nombre}_std20']   = df_extendido[col].rolling(20).std()
    df_extendido[f'{nombre}_zscore']  = (
        df_extendido[col] - df_extendido[f'{nombre}_ma20']
    ) / df_extendido[f'{nombre}_std20']

# 2. Relaciones
df_extendido['wti_brent_spread'] = df_extendido['crudeoil_wti_value'] - df_extendido['crudeoil_brent_value']
df_extendido['yield_curve']      = df_extendido['bono10_value'] - df_extendido['bono2_value']
df_extendido['yield_curve_neg']  = (df_extendido['yield_curve'] < 0).astype(int)
df_extendido['gold_oil_ratio']   = df_extendido['gold_price'] / df_extendido['crudeoil_wti_value']
df_extendido['bono2_slope']      = df_extendido['bono2_value'].diff(5)
df_extendido['bono10_slope']     = df_extendido['bono10_value'].diff(5)

# 3. RSI + técnicos
def rsi(serie, ventana=14):
    delta = serie.diff()
    gain  = delta.clip(lower=0).rolling(ventana).mean()
    loss  = (-delta.clip(upper=0)).rolling(ventana).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

for nombre, col in activos.items():
    df_extendido[f'{nombre}_rsi14']      = rsi(df_extendido[col])
    df_extendido[f'{nombre}_above_ma50'] = (
        df_extendido[col] > df_extendido[col].rolling(50).mean()
    ).astype(int)
    upper = df_extendido[f'{nombre}_ma20'] + 2 * df_extendido[f'{nombre}_std20']
    lower = df_extendido[f'{nombre}_ma20'] - 2 * df_extendido[f'{nombre}_std20']
    df_extendido[f'{nombre}_bb_pos']  = (df_extendido[col] - lower) / (upper - lower + 1e-9)
    df_extendido[f'{nombre}_52w_pct'] = df_extendido[col] / df_extendido[col].rolling(252).max()

# 4. Volatilidad
for nombre, col in activos.items():
    vol = df_extendido[f'{nombre}_logret'].rolling(20).std() * np.sqrt(252)
    df_extendido[f'{nombre}_vol20_ann'] = vol
    df_extendido[f'{nombre}_high_vol']  = (
        vol > vol.expanding().quantile(0.75)
    ).astype(int)
    peak = df_extendido[col].expanding().max()
    df_extendido[f'{nombre}_drawdown']  = (df_extendido[col] - peak) / peak

# Momentum
rets_21d = [f'{n}_ret_21d' for n in activos]
df_extendido['momentum_score'] = df_extendido[rets_21d].apply(
    lambda row: (row - row.mean()) / (row.std() + 1e-9), axis=1
).mean(axis=1)

# Crisis flag
crisis_periods = [
    ('2014-10-01', '2016-02-28'),
    ('2018-10-01', '2019-01-31'),
    ('2020-02-01', '2020-05-31'),
    ('2022-01-01', '2022-10-31'),
]
df_extendido['crisis_flag'] = 0
for start, end in crisis_periods:
    mask = (df_extendido['date'] >= start) & (df_extendido['date'] <= end)
    df_extendido.loc[mask, 'crisis_flag'] = 1

# ── 3. REGÍMENES (CLAVE) ─────────────────────────────────────────────────────
df_extendido['gold_ret_5d']   = df_extendido['gold_price'].pct_change(5)
df_extendido['bono10_ret_5d'] = df_extendido['bono10_value'].pct_change(5)
df_extendido['wti_ret_5d_now'] = df_extendido['crudeoil_wti_value'].pct_change(5)

UMBRAL_CAIDA_FUERTE = -0.05
UMBRAL_SUBIDA       =  0.01

df_extendido['regimen_crecimiento'] = (
    (df_extendido['wti_ret_5d_now'] > UMBRAL_SUBIDA) &
    (df_extendido['bono10_ret_5d']  > UMBRAL_SUBIDA)
).astype(int)

df_extendido['regimen_miedo'] = (
    (df_extendido['gold_ret_5d']    > UMBRAL_SUBIDA) &
    (df_extendido['wti_ret_5d_now'] > UMBRAL_SUBIDA)
).astype(int)

df_extendido['regimen_riesgo'] = (
    df_extendido['wti_ret_5d_now'] < UMBRAL_CAIDA_FUERTE
).astype(int)

df_extendido['regimen_estanflacion'] = (
    (df_extendido['wti_ret_5d_now'] > UMBRAL_SUBIDA) &
    (df_extendido['bono10_ret_5d']  < -UMBRAL_SUBIDA)
).astype(int)

# ── 4. PREDICCIÓN ────────────────────────────────────────────────────────────
fila_hoy = df_extendido[FEATURES].iloc[[-1]]

prob = modelo.predict_proba(fila_hoy)[0, 1]
pred = modelo.predict(fila_hoy)[0]

if prob >= 0.65:   señal = '🟢 FAVORABLE'
elif prob >= 0.50: señal = '🟡 LEVE FAVORABLE'
elif prob >= 0.35: señal = '🟠 LEVE DESFAVORABLE'
else:              señal = '🔴 DESFAVORABLE'

yc = df_extendido['yield_curve'].iloc[-1]


# ==============================
# GAUGE (Market Score)
# ==============================
import plotly.graph_objects as go

score2 = float(prob) * 100

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score2,
    number={'suffix': "%"},
    title={'text': "Market Score"},

    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "black"},

        'steps': [
            {'range': [0, 35], 'color': "#d73027"},
            {'range': [35, 50], 'color': "#fc8d59"},
            {'range': [50, 65], 'color': "#fee08b"},
            {'range': [65, 100], 'color': "#1a9850"}
        ],

        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': score2
        }
    }
))

fig_gauge.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))





# ==============================
# OUTPUT DASHBOARD
# ==============================
st.subheader("📊 Resultado del modelo")

col1, col2 = st.columns([1, 2])

with col1:
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad", f"{prob:.1%}")
    c2.metric("Predicción", "Favorable" if pred == 1 else "Desfavorable")
    c3.metric("Yield Curve", f"{yc:.2f}")

    st.markdown("### Señal")
    st.markdown(f"## {señal}")

    st.markdown("### Regímenes")
    st.write({
        "Crecimiento": int(df_extendido['regimen_crecimiento'].iloc[-1]),
        "Miedo": int(df_extendido['regimen_miedo'].iloc[-1]),
        "Riesgo": int(df_extendido['regimen_riesgo'].iloc[-1]),
        "Estanflación": int(df_extendido['regimen_estanflacion'].iloc[-1]),
    })


# ==============================
# LAYOUT DASHBOARD
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ==============================
# TABLA
# ==============================
st.subheader("Datos")
st.dataframe(panorama_df.tail())