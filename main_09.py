# ==============================
# LIBRERIAS
# ==============================
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from prompts import stronger_prompt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
from pydantic import BaseModel, Field


# ========================================================================================================================
# NOTICIAS
# ========================================================================================================================

# --------------------- Cargar variables de entorno ---------------------
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------- Crear cliente de OpenAI ---------------------
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini"

# --------------------- Cargar dataset ---------------------
DATA_DIR = Path(__file__).resolve().parent
df_news = pd.read_csv(DATA_DIR / "df_news.csv")  # df_news debe tener columna "title"

# --------------------- Pydantic para el resumen global ---------------------
class GlobalInsights(BaseModel):
    preocupaciones: List[str] = Field(default_factory=list, description="Principales riesgos o problemas detectados")
    avances_salud: List[str] = Field(default_factory=list, description="Avances, fusiones o noticias del sector salud")

# --------------------- Función para resumen global ---------------------
def get_global_insights(titles: List[str]) -> GlobalInsights:
    combined_text = "\n".join(titles)
    prompt = f"""
Analiza las siguientes noticias financieras y económicas en español. 
Devuelve un JSON con dos campos:

1. 'preocupaciones': Lista de principales riesgos o problemas detectados.
2. 'avances_salud': Lista de avances, fusiones o noticias importantes relacionadas con el sector salud.

Noticias:
{combined_text}
"""
    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Eres un experto analista financiero y de mercado. Devuelve SOLO un JSON válido con los campos 'preocupaciones' y 'avances_salud'."
            },
            {"role": "user", "content": prompt},
        ],
        response_format=GlobalInsights
    )
    return response.choices[0].message.parsed



# ========================================================================================================================
# PROMPT
# ========================================================================================================================



# ========================================================================================================================
# MODELO
# ========================================================================================================================
_DATA_DIR = Path(__file__).resolve().parent
panorama_df = pd.read_csv(_DATA_DIR / "panorama_df.csv")

st.set_page_config(layout="wide")
st.title("📊 Macro Dashboard")

# ==============================
# LIMPIEZA
# ==============================
panorama_df['date'] = pd.to_datetime(panorama_df['date'])
panorama_df = panorama_df.sort_values('date').reset_index(drop=True)
panorama_df = panorama_df.drop(columns=['Unnamed: 0'])

# ==============================
# KPIs
# ==============================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("WTI", round(panorama_df["crudeoil_wti_value"].iloc[-1], 2))
c2.metric("Brent", round(panorama_df["crudeoil_brent_value"].iloc[-1], 2))
c3.metric("Bono 10Y", round(panorama_df["bono10_value"].iloc[-1], 2))
c4.metric("Bono 2Y", round(panorama_df["bono2_value"].iloc[-1], 2))
c5.metric("Oro", round(panorama_df["gold_price"].iloc[-1], 2))

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

#fig4 = make_subplots()
#fig4.add_trace(go.Scatter(x=panorama_df["date"], y=panorama_df["bono10_value"], name="10Y"))
#fig4.add_trace(go.Scatter(x=panorama_df["date"], y=panorama_df["bono2_value"], name="2Y"))

# ==============================
# TARGET Y REGIMENES
# ==============================
HORIZONTE = 5

panorama_df['wti_ret_forward5'] = (
    panorama_df['crudeoil_wti_value'].shift(-HORIZONTE) /
    panorama_df['crudeoil_wti_value'] - 1
)

panorama_df['gold_ret_5d']   = panorama_df['gold_price'].pct_change(5)
panorama_df['bono10_ret_5d'] = panorama_df['bono10_value'].pct_change(5)
panorama_df['wti_ret_5d_now'] = panorama_df['crudeoil_wti_value'].pct_change(5)

UMBRAL_CAIDA_FUERTE = -0.05
UMBRAL_SUBIDA = 0.01

panorama_df['regimen_crecimiento'] = (
    (panorama_df['wti_ret_5d_now'] > UMBRAL_SUBIDA) &
    (panorama_df['bono10_ret_5d']  > UMBRAL_SUBIDA)
).astype(int)

panorama_df['regimen_miedo'] = (
    (panorama_df['gold_ret_5d'] > UMBRAL_SUBIDA) &
    (panorama_df['wti_ret_5d_now'] > UMBRAL_SUBIDA)
).astype(int)

panorama_df['regimen_riesgo'] = (
    panorama_df['wti_ret_5d_now'] < UMBRAL_CAIDA_FUERTE
).astype(int)

panorama_df['mercado_favorable'] = (
    (panorama_df['wti_ret_forward5'] > 0) &
    (panorama_df['yield_curve'] > 0) &
    (panorama_df['regimen_riesgo'] == 0) &
    (panorama_df['regimen_miedo'] == 0)
).astype(int)


panorama_df['regimen_estanflacion'] = (
    (panorama_df['wti_ret_5d_now'] > UMBRAL_SUBIDA) &
    (panorama_df['bono10_ret_5d']  < -UMBRAL_SUBIDA)
).astype(int)

def asignar_regimen_multiclase(row):
    if row['regimen_riesgo'] == 1:
        return 0
    elif row['regimen_crecimiento'] == 1 and row['regimen_miedo'] == 0:
        return 2
    elif row['regimen_miedo'] == 1:
        return 3
    else:
        return 1

panorama_df['regimen_actual'] = panorama_df.apply(asignar_regimen_multiclase, axis=1)

panorama_df = panorama_df.dropna()

# ==============================
# MODELO
# ==============================
FEATURES = [
    "bono10_rsi14","gold_ret_21d","wti_ret_5d","wti_ret_21d",
    "gold_52w_pct","wti_rsi14","wti_vol20_ann","wti_52w_pct",
    "wti_drawdown","gold_vol20_ann","gold_drawdown",
    "wti_brent_spread","bono10_vol20_ann","gold_oil_ratio",
    "yield_curve_neg","regimen_miedo","bono2_drawdown","gold_price"
]

X = panorama_df[FEATURES]
y = panorama_df['mercado_favorable']

split = int(len(X)*0.8)
modelo = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42
)

modelo.fit(X.iloc[:split], y.iloc[:split])

# ==============================
# INPUT USUARIO
# ==============================
st.markdown("## 🔮 Predicción")

c1,c2,c3,c4,c5 = st.columns(5)

wti = c1.text_input("WTI")
brent = c2.text_input("Brent")
bono2 = c3.text_input("Bono 2Y")
bono10 = c4.text_input("Bono 10Y")
gold = c5.text_input("Gold")

calcular = st.button("Calcular")

# ==============================
# FUNCIONES
# ==============================
def rsi(serie, n=14):
    delta = serie.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0,np.nan)
    return 100 - (100/(1+rs))

# ==============================
# PREDICCION CORRECTA
# ==============================
# ==============================
# PREDICCION CORRECTA
# ==============================
# ── Constantes ──────────────────────────────────────────────────────────────
#UMBRAL_SUBIDA       = 0.02   # +2% en 5 días
#UMBRAL_CAIDA_FUERTE = 0.03   # 3% caída en 5 días (se usa como negativo)

# ── Función de régimen multiclase ───────────────────────────────────────────
def asignar_regimen_multiclase(row):
    """
    0 = Riesgo       → WTI cae fuerte
    1 = Neutro       → ninguna condición dominante
    2 = Crecimiento  → WTI sube + bono10 sube
    3 = Miedo        → Gold sube + WTI sube
    """
    wti   = row['wti_ret_5d_now']
    b10   = row['bono10_ret_5d']
    gold  = row['gold_ret_5d']

    if wti < -UMBRAL_CAIDA_FUERTE:
        return 0  # Riesgo — prioridad máxima
    elif (wti > UMBRAL_SUBIDA) and (b10 > UMBRAL_SUBIDA):
        return 2  # Crecimiento
    elif (gold > UMBRAL_SUBIDA) and (wti > UMBRAL_SUBIDA):
        return 3  # Miedo
    else:
        return 1  # Neutro

# ── Bloque de predicción completo ───────────────────────────────────────────
if calcular:

    if not all([wti, brent, bono2, bono10, gold]):
        st.error("⚠️ Ingresa los 5 valores")
        st.stop()

    try:
        wti, brent, bono2, bono10, gold = map(float, [wti, brent, bono2, bono10, gold])
    except:
        st.error("⚠️ Todos deben ser numéricos")
        st.stop()

    # ── Cargar CSV original completo ─────────────────────────────────────────
    df = pd.read_csv(_DATA_DIR / "panorama_df.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # ── Verificar historia suficiente ────────────────────────────────────────
    cols_base = ['crudeoil_wti_value', 'crudeoil_brent_value',
                 'bono2_value', 'bono10_value', 'gold_price']
    filas_validas = df[cols_base].dropna().shape[0]
    if filas_validas < 260:  # necesitamos 252 para rolling 52w
        st.error(f"⚠️ Solo hay {filas_validas} filas con datos. Se necesitan al menos 260.")
        st.stop()

    # ── Inyectar valores del usuario en la última fila ───────────────────────
    df.loc[df.index[-1], 'crudeoil_wti_value']  = wti
    df.loc[df.index[-1], 'crudeoil_brent_value'] = brent
    df.loc[df.index[-1], 'bono2_value']           = bono2
    df.loc[df.index[-1], 'bono10_value']          = bono10
    df.loc[df.index[-1], 'gold_price']            = gold

    # ── Recalcular todos los features ────────────────────────────────────────

    # Retornos
    df['wti_ret_5d']      = df['crudeoil_wti_value'].pct_change(5)
    df['wti_ret_21d']     = df['crudeoil_wti_value'].pct_change(21)
    df['gold_ret_21d']    = df['gold_price'].pct_change(21)
    df['gold_ret_5d']     = df['gold_price'].pct_change(5)
    df['bono10_ret_5d']   = df['bono10_value'].pct_change(5)
    df['wti_ret_5d_now']  = df['crudeoil_wti_value'].pct_change(5)

    # Spreads y ratios
    df['wti_brent_spread'] = df['crudeoil_wti_value'] - df['crudeoil_brent_value']
    df['yield_curve']      = df['bono10_value'] - df['bono2_value']
    df['yield_curve_neg']  = (df['yield_curve'] < 0).astype(int)
    df['gold_oil_ratio']   = df['gold_price'] / df['crudeoil_wti_value']

    # RSI
    df['wti_rsi14']    = rsi(df['crudeoil_wti_value'])
    df['bono10_rsi14'] = rsi(df['bono10_value'])

    # 52-week percentile
    df['wti_52w_pct']  = df['crudeoil_wti_value'] / df['crudeoil_wti_value'].rolling(252).max()
    df['gold_52w_pct'] = df['gold_price'] / df['gold_price'].rolling(252).max()

    # Volatilidad anualizada
    df['wti_vol20_ann']    = np.log(df['crudeoil_wti_value'] / df['crudeoil_wti_value'].shift(1)).rolling(20).std() * np.sqrt(252)
    df['gold_vol20_ann']   = np.log(df['gold_price']         / df['gold_price'].shift(1)        ).rolling(20).std() * np.sqrt(252)
    df['bono10_vol20_ann'] = np.log(df['bono10_value']       / df['bono10_value'].shift(1)      ).rolling(20).std() * np.sqrt(252)

    # Drawdowns
    df['wti_drawdown']   = (df['crudeoil_wti_value'] - df['crudeoil_wti_value'].expanding().max()) / df['crudeoil_wti_value'].expanding().max()
    df['gold_drawdown']  = (df['gold_price']         - df['gold_price'].expanding().max()         ) / df['gold_price'].expanding().max()
    df['bono2_drawdown'] = (df['bono2_value']         - df['bono2_value'].expanding().max()        ) / df['bono2_value'].expanding().max()

    # ── Regímenes binarios ───────────────────────────────────────────────────
    df['regimen_crecimiento'] = (
        (df['wti_ret_5d_now'] > UMBRAL_SUBIDA) &
        (df['bono10_ret_5d']  > UMBRAL_SUBIDA)
    ).astype(int)

    df['regimen_miedo'] = (
        (df['gold_ret_5d']    > UMBRAL_SUBIDA) &
        (df['wti_ret_5d_now'] > UMBRAL_SUBIDA)
    ).astype(int)

    df['regimen_riesgo'] = (
        df['wti_ret_5d_now'] < -UMBRAL_CAIDA_FUERTE  # ← negativo, corrección clave
    ).astype(int)

    df['regimen_estanflacion'] = (
        (df['wti_ret_5d_now'] > UMBRAL_SUBIDA) &
        (df['bono10_ret_5d']  < -UMBRAL_SUBIDA)
    ).astype(int)

    # ── Régimen multiclase ───────────────────────────────────────────────────
    df['regimen_actual'] = df.apply(asignar_regimen_multiclase, axis=1)

    # ── Extraer fila de predicción ───────────────────────────────────────────
    fila = df[FEATURES].iloc[[-1]]

    if fila.isna().sum().sum() > 0:
        nan_cols = fila.columns[fila.isna().any()].tolist()
        st.error(f"⚠️ Columnas con NaN: {nan_cols}")
        st.stop()

    prob = modelo.predict_proba(fila)[0, 1]
    pred = modelo.predict(fila)[0]

    # ── Señal ────────────────────────────────────────────────────────────────
    if prob >= 0.65:
        señal = '🟢 FAVORABLE'
    elif prob >= 0.50:
        señal = '🟡 LEVE FAVORABLE'
    elif prob >= 0.35:
        señal = '🟠 LEVE DESFAVORABLE'
    else:
        señal = '🔴 DESFAVORABLE'

    yc            = df['yield_curve'].iloc[-1]
    regimen_map   = {0: "⚠️ Riesgo", 1: "😐 Neutro", 2: "📈 Crecimiento", 3: "😨 Miedo"}
    regimen_label = regimen_map[int(df['regimen_actual'].iloc[-1])]

    # ── Gauge ────────────────────────────────────────────────────────────────
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
                {'range': [0,   35], 'color': "#d73027"},
                {'range': [35,  50], 'color': "#fc8d59"},
                {'range': [50,  65], 'color': "#fee08b"},
                {'range': [65, 100], 'color': "#1a9850"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score2
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))

# ========================================================================================================================
# INTERFAZ
# ========================================================================================================================
    # ── Output ───────────────────────────────────────────────────────────────
    st.subheader("📊 Resultado del modelo")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Probabilidad",  f"{prob:.1%}")
        c2.metric("Predicción",    "Favorable" if pred == 1 else "Desfavorable")
        c3.metric("Yield Curve",   f"{yc:.2f}")
        c4.metric("Régimen",       regimen_label)

        st.markdown("### Señal")
        st.markdown(f"## {señal}")

# ==============================
# GRAFICOS
# ==============================
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

# --------------------- noticias ---------------------
st.title("📊 Resumen Global de Noticias Financieras del Sector Salud")
#st.write("Genera un análisis consolidado de todas las noticias, con riesgos y avances/fusiones en el sector salud.")

if st.button("Generar Resumen"):
    with st.spinner("Analizando todas las noticias..."):
        insights = get_global_insights(df_news["title"].tolist())

        # Dos columnas para mostrar la información lado a lado
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏥 Avances y Fusiones en Sector Salud")
            if insights.avances_salud:
                for a in insights.avances_salud:
                    st.write(f"- {a}")
            else:
                st.write("No se detectaron avances o fusiones en el sector salud.")

        with col2:
            st.subheader("⚠️ Principales Preocupaciones")
            if insights.preocupaciones:
                for p in insights.preocupaciones:
                    st.write(f"- {p}")
            else:
                st.write("No se detectaron preocupaciones destacables.")




# --------------------- Chat al final de la interfaz: PROMPT ---------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)
model_openai = "gpt-5.4-mini"

# --- Estado del chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# --- Botón para abrir/cerrar el chat ---
col1, col2 = st.columns([6, 1])
with col2:
    if not st.session_state.chat_open:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.chat_open = True
            st.rerun()
    else:
        if st.button("✕ Cerrar", use_container_width=True):
            st.session_state.chat_open = False
            st.session_state.messages = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]
            st.rerun()

# --- Interfaz del chat (solo si está abierto) ---
if st.session_state.chat_open:
    st.divider()

    # Mostrar historial de mensajes
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribe tu mensaje aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Construir conversación con el system prompt
        conversation = [{"role": "system", "content": stronger_prompt}]
        conversation.extend(
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        )

        # Respuesta del asistente con streaming
        with st.chat_message("assistant"):
            stream = client_openai.chat.completions.create(
                model=model_openai,
                messages=conversation,
                stream=True
            )
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})
