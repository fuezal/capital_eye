
# ========================================================================================================================
# CODIGO COMPLETO: Version final que integra todos los codigos
# ========================================================================================================================

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
import time
import requests
import plotly.express as px


# Librerías para noticias
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer



# ==============================
# CONFIGURACIÓN INICIAL
# ==============================
st.set_page_config(layout="wide")
load_dotenv(override=True)

# ========================================================================================================================
# LOGO
# ========================================================================================================================

# Configuración de la página
st.set_page_config(layout="wide")

# Barra superior con fondo negro solo con el logo
logo_bar = """
<div style="background-color: #000000; padding: 10px 20px; display: flex; align-items: center; justify-content: flex-start;">
    <!-- Logo SVG -->
    <svg width="120" height="50" viewBox="0 0 420 100">
        <circle cx="50" cy="50" r="28" stroke="#1E40FF" stroke-width="6" fill="none"/>
        <circle cx="50" cy="50" r="18" fill="#1E40FF"/>
        <text x="100" y="60" font-family="Arial" font-size="40" style="font-weight:bold" fill="#38D6C4">
        CAPITAL
        </text>
        <text x="270" y="60" font-family="Arial" font-size="40" style="font-weight:bold" fill="#D1D5DB">
        EYE
        </text>
    </svg>
</div>
"""
# Mostrar barra superior
st.markdown(logo_bar, unsafe_allow_html=True)

# Contenido del dashboard
#st.write("Herramienta financiera")

# ==============================
# NAVEGACIÓN
# ==============================
st.sidebar.title("📌 Navegación")

seccion = st.sidebar.selectbox(
    "Selecciona una sección",
    [
        "🌎 Panorama del mercado",
        "📊 Razones financieras",
        "📰 Noticias financieras",
        "💬 Chat de apoyo"
    ],
    index=0
)

# ========================================================================================================================
# 🌎 PANORAMA DEL MERCADO
# ========================================================================================================================
if seccion == "🌎 Panorama del mercado":

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
    _DATA_DIR = Path(__file__).resolve().parent
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


# ========================================================================================================================
# 📰 NOTICIAS
# ========================================================================================================================
elif seccion == "📰 Noticias financieras":

    # --------------------- noticias ---------------------
    # Descargar léxico si no existe
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

    # --------------------- UI ---------------------
    st.title("📊 Noticias Financieras e Internacionales")

    user_ticker = st.text_input("Ingresa el ticker (ej: AAPL, TSLA, JNJ)", value="JNJ")

    # --------------------- OpenAI config ---------------------
    load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL_NAME = "gpt-4o-mini"

    # --------------------- Pydantic ---------------------
    class GlobalInsights(BaseModel):
        preocupaciones: List[str] = Field(default_factory=list)
        avances: List[str] = Field(default_factory=list)

    # --------------------- Función OpenAI ---------------------
    def get_global_insights(titles: List[str]) -> GlobalInsights:
        combined_text = "\n".join(titles)

        prompt = f"""
    Analiza las siguientes noticias financieras y económicas en español. 
    Devuelve un JSON con dos campos:

    1. 'preocupaciones': Lista de principales riesgos o problemas detectados.
    2. 'avances': Lista de avances, fusiones o noticias importantes.

    Noticias:
    {combined_text}
    """

        response = client.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto analista financiero. Devuelve SOLO JSON válido en español."
                },
                {"role": "user", "content": prompt},
            ],
            response_format=GlobalInsights
        )

        return response.choices[0].message.parsed

    # --------------------- BOTÓN PRINCIPAL ---------------------
    if st.button("Generar Resumen Global"):

        if not user_ticker:
            st.warning("Por favor ingresa un ticker válido.")
            st.stop()

        ticker = user_ticker.upper()

        # --------------------- Scraping ---------------------
        finviz_url = "https://finviz.com/quote.ashx?t="
        parsed_data = []

        try:
            url = finviz_url + ticker
            req = Request(url=url, headers={"user-agent": "app"})
            response = urlopen(req)

            html = BeautifulSoup(response, 'html.parser')
            news_table = html.find(id="news-table")

            if news_table is None:
                st.error(f"No se encontraron noticias para {ticker}")
                st.stop()

            for row in news_table.find_all('tr'):
                if row.a is not None:
                    title = row.a.text.strip()
                    timestamp = row.td.text.split()

                    if len(timestamp) == 1:
                        date = None
                        time_val = timestamp[0]
                    else:
                        date = timestamp[0].lower()
                        time_val = timestamp[1]

                    parsed_data.append([ticker, date, time_val, title])

        except Exception as e:
            st.error(f"Error al obtener datos: {e}")
            st.stop()

        # --------------------- DataFrame (interno, NO se muestra) ---------------------
        df_news = pd.DataFrame(parsed_data, columns=["ticker", "date", "time", "title"])

        # --------------------- Sentimiento ---------------------
        vader = SentimentIntensityAnalyzer()

        df_news['compound'] = df_news['title'].apply(lambda x: vader.polarity_scores(x)['compound'])

        df_news['sentiment'] = np.where(
            df_news['compound'] > 0, 'POS',
            np.where(df_news['compound'] < 0, 'NEG', 'NEU')
        )

        df_news.drop(columns=['compound'], inplace=True)

        # --------------------- Análisis con IA ---------------------
        st.subheader(f"📊 Análisis para {ticker}")

        with st.spinner("Analizando noticias..."):
            insights = get_global_insights(df_news["title"].tolist())

            st.subheader("⚠️ Principales Preocupaciones")
            if insights.preocupaciones:
                for p in insights.preocupaciones:
                    st.write(f"- {p}")
            else:
                st.write("No se detectaron preocupaciones.")

            st.subheader("🚀 Avances o Progreso")
            if insights.avances:
                for a in insights.avances:
                    st.write(f"- {a}")
            else:
                st.write("No se detectaron avances.")

# ========================================================================================================================
# 💬 CHAT
# ========================================================================================================================
elif seccion == "💬 Chat de apoyo":

    st.write("Dar click en el boton")

    with st.expander("💬 Chat de Apoyo — ¿Qué encontrarás aquí?"):
        st.markdown("""
    **Asistente de análisis financiero integral**

    **¿Qué puedes hacer?**
    - 🏢 Modelo de negocio de empresas
    - 📊 Razones financieras (análisis independiente)
    - 🌐 Indicadores macro (tasas, petróleo, oro)
    - 🔗 Conectar empresa + macro  

    **¿Cómo responde?**
    - Separa: Empresa | Finanzas | Macro  
    - Explica: 👉 causa → efecto  
    - Incluye conclusión + pregunta guía  

    **Importante**
    - ❌ No da recomendaciones de compra/venta  
    - ❌ No responde temas fuera de finanzas  

    **Empieza con:**
    - “Analiza [empresa]”  
    - “Explícame el entorno actual”
    """)


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






#========================================================================================================================
# 📊 RAZONES FINANCIERAS
# ========================================================================================================================
elif seccion == "📊 Razones financieras":

    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") 

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


    # ========================================================================================================================
    # INTERFAZ: RAZONES FINANCIERAS
    # ========================================================================================================================

    st.title("📊 Razones Financieras:")

    # ── Parámetro de Ticker ───────────────────────────────────────────────
    ticker_input = st.text_input("Ingresa el ticker de alguna empresa del mercado estadounidense. Ejemplo:JNJ,KO,COST", value="JNJ").upper()
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
            df_loaded = load_data(st.session_state["ticker"])
            if df_loaded.empty:
                st.error("❌ Ticker inválido o sin datos disponibles.")
                st.stop()
            else:
                st.session_state["df"] = df_loaded

    # Mostrar mensaje si no hay datos
    if st.session_state["df"].empty:
        st.info("Ingresa un ticker arriba y presiona **Cargar datos** para ver los ratios financieros.")
    else:
        df = st.session_state["df"]
        ticker = st.session_state["ticker"]
        st.success(f"**{ticker}** — {len(df)} años cargados")

        # Diccionarios de indicadores
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

        # Selección de indicadores
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            label1 = st.selectbox("Indicador (Ratio)", list(indicadores.keys()), key="g1")
        with col_sel2:
            label2 = st.selectbox("Indicador (Crecimiento %)", list(indicadores_growth.keys()), key="g2")

        # Gráficos
        fig1 = bar_plot1(df, indicadores[label1], label1)
        fig2 = bar_plot2(df, indicadores_growth[label2], label2)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Ratios Financieros")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.subheader("📊 Crecimiento YoY (%)")
            st.plotly_chart(fig2, use_container_width=True)