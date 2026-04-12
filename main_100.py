# ========================================================================================================================
# CODIGO INDIVIDUAL: Modelo del panorama del mercado (VERSIÓN CORREGIDA)
# ========================================================================================================================

# ==============================
# LIBRERIAS
# ==============================
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================
# CONFIGURACIÓN
# ==============================
st.set_page_config(layout="wide")
load_dotenv(override=True)

_DATA_DIR = Path(__file__).resolve().parent

# ==============================
#
# ==============================










# ==============================
# FUNCIONES
# ==============================
def rsi(serie, n=14):
    delta = serie.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df):
    # Retornos
    df['wti_ret_5d'] = df['crudeoil_wti_value'].pct_change(5)
    df['wti_ret_21d'] = df['crudeoil_wti_value'].pct_change(21)
    df['gold_ret_21d'] = df['gold_price'].pct_change(21)
    df['gold_ret_5d'] = df['gold_price'].pct_change(5)
    df['bono10_ret_5d'] = df['bono10_value'].pct_change(5)
    df['wti_ret_5d_now'] = df['crudeoil_wti_value'].pct_change(5)

    # Spreads
    df['wti_brent_spread'] = df['crudeoil_wti_value'] - df['crudeoil_brent_value']
    df['yield_curve'] = df['bono10_value'] - df['bono2_value']
    df['yield_curve_neg'] = (df['yield_curve'] < 0).astype(int)
    df['gold_oil_ratio'] = df['gold_price'] / df['crudeoil_wti_value']

    # RSI
    df['wti_rsi14'] = rsi(df['crudeoil_wti_value'])
    df['bono10_rsi14'] = rsi(df['bono10_value'])

    # Percentiles
    df['wti_52w_pct'] = df['crudeoil_wti_value'] / df['crudeoil_wti_value'].rolling(252).max()
    df['gold_52w_pct'] = df['gold_price'] / df['gold_price'].rolling(252).max()

    # Volatilidad
    df['wti_vol20_ann'] = np.log(df['crudeoil_wti_value'] / df['crudeoil_wti_value'].shift(1)).rolling(20).std() * np.sqrt(252)
    df['gold_vol20_ann'] = np.log(df['gold_price'] / df['gold_price'].shift(1)).rolling(20).std() * np.sqrt(252)
    df['bono10_vol20_ann'] = np.log(df['bono10_value'] / df['bono10_value'].shift(1)).rolling(20).std() * np.sqrt(252)
    df['brent_vol20_ann'] = np.log(df['crudeoil_brent_value'] / df['crudeoil_brent_value'].shift(1)).rolling(20).std() * np.sqrt(252)
    df['bono2_vol20_ann'] = np.log(df['bono2_value'] / df['bono2_value'].shift(1)).rolling(20).std() * np.sqrt(252)

    # Drawdowns
    df['wti_drawdown'] = (df['crudeoil_wti_value'] - df['crudeoil_wti_value'].expanding().max()) / df['crudeoil_wti_value'].expanding().max()
    df['gold_drawdown'] = (df['gold_price'] - df['gold_price'].expanding().max()) / df['gold_price'].expanding().max()
    df['bono2_drawdown'] = (df['bono2_value'] - df['bono2_value'].expanding().max()) / df['bono2_value'].expanding().max()

    return df


def asignar_regimen(row, UMBRAL_SUBIDA=0.01, UMBRAL_CAIDA=0.05):
    wti = row['wti_ret_5d_now']
    b10 = row['bono10_ret_5d']
    gold = row['gold_ret_5d']

    if wti < -UMBRAL_CAIDA:
        return 0
    elif (wti > UMBRAL_SUBIDA) and (b10 > UMBRAL_SUBIDA):
        return 2
    elif (gold > UMBRAL_SUBIDA) and (wti > UMBRAL_SUBIDA):
        return 3
    else:
        return 1


# ==============================
# CARGA DATA
# ==============================
panorama_df = pd.read_csv(_DATA_DIR / "panorama_df.csv")
panorama_df['date'] = pd.to_datetime(panorama_df['date'])
panorama_df = panorama_df.sort_values('date').reset_index(drop=True)
panorama_df = panorama_df.drop(columns=['Unnamed: 0'], errors='ignore')

panorama_df = build_features(panorama_df)

# TARGET
HORIZONTE = 5
panorama_df['wti_ret_forward5'] = panorama_df['crudeoil_wti_value'].shift(-HORIZONTE) / panorama_df['crudeoil_wti_value'] - 1

panorama_df['regimen_actual'] = panorama_df.apply(asignar_regimen, axis=1)

panorama_df['mercado_favorable'] = (
    (panorama_df['wti_ret_forward5'] > 0) &
    (panorama_df['yield_curve'] > 0) &
    (panorama_df['regimen_actual'] != 0) &
    (panorama_df['regimen_actual'] != 3)
).astype(int)

# ==============================
# MODELO
# ==============================
FEATURES = [
    "bono10_rsi14","gold_ret_21d","wti_ret_5d","wti_ret_21d",
    "gold_52w_pct","wti_rsi14","wti_vol20_ann","wti_52w_pct",
    "wti_drawdown","gold_vol20_ann","gold_drawdown",
    "wti_brent_spread","bono10_vol20_ann","gold_oil_ratio",
    "yield_curve_neg","bono2_drawdown","gold_price"
]

panorama_df = panorama_df.dropna(subset=FEATURES + ['mercado_favorable'])

X = panorama_df[FEATURES]
y = panorama_df['mercado_favorable']

split = int(len(X) * 0.8)

modelo = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42
)

modelo.fit(X.iloc[:split], y.iloc[:split])

# ==============================
# UI
# ==============================
st.title("📊 Macro Dashboard")

#c1, c2, c3, c4, c5 = st.columns(5)
#c1.metric("WTI", round(df["crudeoil_wti_value"].iloc[-1], 2))
#c2.metric("Brent", round(df["crudeoil_brent_value"].iloc[-1], 2))
#c3.metric("Bono 10Y", round(df["bono10_value"].iloc[-1], 2))
#c4.metric("Bono 2Y", round(df["bono2_value"].iloc[-1], 2))
#c5.metric("Oro", round(df["gold_price"].iloc[-1], 2))

st.markdown("---")

# ==============================
# INPUT
# ==============================
st.subheader("🔮 Predicción")

c1, c2, c3, c4, c5 = st.columns(5)

wti = c1.number_input("WTI", value=float(panorama_df["crudeoil_wti_value"].iloc[-1]))
brent = c2.number_input("Brent", value=float(panorama_df["crudeoil_brent_value"].iloc[-1]))
bono2 = c3.number_input("Bono 2Y", value=float(panorama_df["bono2_value"].iloc[-1]))
bono10 = c4.number_input("Bono 10Y", value=float(panorama_df["bono10_value"].iloc[-1]))
gold = c5.number_input("Gold", value=float(panorama_df["gold_price"].iloc[-1]))

if st.button("Calcular"):

    df_pred = panorama_df.copy()

    df_pred.loc[df_pred.index[-1], 'crudeoil_wti_value'] = wti
    df_pred.loc[df_pred.index[-1], 'crudeoil_brent_value'] = brent
    df_pred.loc[df_pred.index[-1], 'bono2_value'] = bono2
    df_pred.loc[df_pred.index[-1], 'bono10_value'] = bono10
    df_pred.loc[df_pred.index[-1], 'gold_price'] = gold

    df_pred = build_features(df_pred)
    df_pred['regimen_actual'] = df_pred.apply(asignar_regimen, axis=1)

    fila = df_pred[FEATURES].iloc[[-1]]

    prob = modelo.predict_proba(fila)[0, 1]
    pred = modelo.predict(fila)[0]

    # ==============================
    # KPIs
    # ==============================
    score = prob * 100
    yc = df_pred['yield_curve'].iloc[-1]

    # Régimen
    regimen_map = {
        0: "⚠️ Riesgo",
        1: "😐 Neutro",
        2: "📈 Crecimiento",
        3: "😨 Miedo"
    }
    regimen = regimen_map[int(df_pred['regimen_actual'].iloc[-1])]

    # Señal
    if prob >= 0.65:
        señal = "🟢 FAVORABLE"
    elif prob >= 0.50:
        señal = "🟡 LEVE FAVORABLE"
    elif prob >= 0.35:
        señal = "🟠 LEVE DESFAVORABLE"
    else:
        señal = "🔴 DESFAVORABLE"

    # ==============================
    # GAUGE CON COLORES
    # ==============================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "%"},
        title={'text': "Market Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 35], 'color': "#d73027"},   # rojo
                {'range': [35, 50], 'color': "#fc8d59"},  # naranja
                {'range': [50, 65], 'color': "#fee08b"},  # amarillo
                {'range': [65, 100], 'color': "#1a9850"}  # verde
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'value': score
            }
        }
    ))

    fig.update_layout(height=300)


    # ==============================
    # VOLATILIDADES ACTUALES
    # ==============================
    vol_wti_kpi  = df_pred['wti_vol20_ann'].iloc[-1]
    vol_brent_kpi  = df_pred['brent_vol20_ann'].iloc[-1]
    vol_bono2_kpi  = df_pred['bono2_vol20_ann'].iloc[-1]
    vol_bono10_kpi = df_pred['bono10_vol20_ann'].iloc[-1]
    vol_gold_kpi   = df_pred['gold_vol20_ann'].iloc[-1]


    # ==============================
    # OUTPUT UI
    # ==============================

    st.subheader("📊 Resultado del modelo")

    col1, col2 = st.columns([1, 1.1])

    # ==============================
    # GAUGE
    # ==============================
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # TABLA KPI PRO
    # ==============================
    with col2:

        # Separación lógica
        df_kpis = pd.DataFrame({
            "Indicador": [
                "🎯 Probabilidad",
                "📌 Predicción",
                "📈 Yield Curve",
                "🌍 Régimen",
                "🚦 Señal",
                "⚡ Vol WTI",
                "⚡ Vol Brent",
                "⚡ Vol 2Y",
                "⚡ Vol 10Y",
                "⚡ Vol Oro"
            ],
            "Valor": [
                f"{prob:.1%}",
                "Favorable" if pred == 1 else "Desfavorable",
                f"{yc:.2f}",
                regimen,
                señal,
                f"{vol_wti_kpi:.1%}",
                f"{vol_brent_kpi:.1%}",
                f"{vol_bono2_kpi:.1%}",
                f"{vol_bono10_kpi:.1%}",
                f"{vol_gold_kpi:.1%}"
            ]
        })

        # ==============================
        # COLOR PROFESIONAL
        # ==============================
        def color_fila(row):
            val = str(row["Valor"])

            if "🟢" in val:
                return ["background-color: #0f5132; color: #d1e7dd"] * 2
            elif "🟡" in val:
                return ["background-color: #664d03; color: #fff3cd"] * 2
            elif "🟠" in val:
                return ["background-color: #7f2704; color: #ffddb5"] * 2
            elif "🔴" in val:
                return ["background-color: #58151c; color: #f8d7da"] * 2
            elif row["Indicador"] == "":
                return ["background-color: black; color: black"] * 2
            else:
                return ["background-color: #111111; color: white"] * 2

        styled_table = (
            df_kpis.style
            .apply(color_fila, axis=1)
            .set_properties(**{
                "font-size": "15px",
                "text-align": "center",
                "border": "1px solid #222"
            })
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#000000"),
                        ("color", "white"),
                        ("font-size", "16px"),
                        ("text-align", "center"),
                        ("border", "1px solid #333")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("padding", "8px"),
                        ("border", "1px solid #222")
                    ]
                }
            ])
        )

        st.markdown("### 📊 Market Snapshot")
        st.dataframe(styled_table, use_container_width=True, height=420)



#graficos

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

col1,col2 = st.columns(2)
col1.plotly_chart(fig3, use_container_width=True)
col2.plotly_chart(fig4, use_container_width=True)

panorama_df3 = pd.read_csv(_DATA_DIR / "panorama_df.csv")

cols = ["date", "crudeoil_wti_value", "crudeoil_brent_value", 
        "bono10_value", "bono2_value", "gold_price", "yield_curve"]

rename_dict = {
    "date": "Fecha",
    "crudeoil_wti_value": "Petróleo WTI",
    "crudeoil_brent_value": "Petróleo Brent",
    "bono10_value": "Bono 10 años",
    "bono2_value": "Bono 2 años",
    "gold_price": "Precio del Oro",
    "yield_curve": "Curva de Rendimiento"
}

df_display = panorama_df3[cols].rename(columns=rename_dict)

st.dataframe(df_display.tail(), hide_index=True)