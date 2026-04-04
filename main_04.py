import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import requests
from typing import List, Optional
from pydantic import BaseModel, Field

# Cargar variables de entorno
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API")

# Crear cliente de OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Modelo de OpenAI
MODEL_NAME = "gpt-4o-mini"

# --------------------- Funciones ---------------------
def render_transcript(d: dict) -> str:
    header_info = f"Symbol: {d.get('symbol','')} | Quarter: {d.get('quarter','')}"
    transcript_lines = [header_info, "TRANSCRIPT:"]
    for index, transcript_entry in enumerate(d.get("transcript", [])):
        speaker_name = transcript_entry.get("speaker", "Unknown")
        speaker_title = transcript_entry.get("title")
        utterance_text = (transcript_entry.get("content") or "").strip()
        speaker_tag = f"{speaker_name} ({speaker_title})" if speaker_title else speaker_name
        transcript_lines.append(f"[{index:04d}] {speaker_tag}: {utterance_text}")
    return "\n".join(transcript_lines)

class EarningsCallInsights(BaseModel):
    symbol: Optional[str] = Field(description="Ticker de la compañía, ej. AAPL, AMZN, MSFT, etc")
    sentiment: Optional[str] = Field(description="Sentimiento general: Muy bajista / Bajista / Alcista / Muy alcista")
    summary: str = Field(description="Resumen ejecutivo de la llamada en español (párrafo de 3 líneas)")
    key_topics: List[str] = Field(default_factory=list, description="Temas estratégicos principales, máximo 4 palabras cada uno")
    guidance: List[str] = Field(default_factory=list, description="Guías o proyecciones futuras mencionadas, en español")
    numeric_highlights: List[str] = Field(default_factory=list, description="Cifras o métricas clave reportadas, en español")
    risks: List[str] = Field(default_factory=list, description="Riesgos explícitos o implícitos discutidos, en español")
    catalysts: List[str] = Field(default_factory=list, description="Catalizadores futuros o eventos relevantes, en español")
    analyst_questions: List[str] = Field(default_factory=list, description="Top 3 preguntas destacadas de analistas, en español")
    unanswered_topics: List[str] = Field(default_factory=list, description="Temas abiertos o sin respuesta clara, en español")
    bullish_points: List[str] = Field(default_factory=list, description="Tesis alcistas derivadas de la llamada, en español")
    bearish_points: List[str] = Field(default_factory=list, description="Tesis bajistas derivadas de la llamada, en español")
    red_flags: List[str] = Field(default_factory=list, description="Alertas o señales negativas detectadas, en español")
    emotion: Optional[str] = Field(description="Emoción general: Optimismo / Incertidumbre / Preocupación / Entusiasmo / Frustración")

def get_earnings_call_insights(call_transcripts: dict) -> EarningsCallInsights:
    transcript_text = render_transcript(call_transcripts)
    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Eres un experto Analista Financiero. Devuelve SOLO un JSON válido que siga exactamente el esquema de EarningsCallInsights. Salidas en español."},
            {"role": "user", "content": transcript_text},
        ],
        response_format=EarningsCallInsights,
    )
    insights = response.choices[0].message.parsed
    return insights.model_dump()

def get_call_transcripts(ticker: str, quarter: str) -> dict:
    url = f'https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}&quarter={quarter}&apikey={ALPHAVANTAGE_API_KEY}'
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error al obtener la transcripción de AlphaVantage.")
        return {}
    call_transcripts = response.json()
    if not call_transcripts.get("transcript"):
        st.warning("No se encontró transcripción para este ticker/trimestre.")
        return {}
    insights = get_earnings_call_insights(call_transcripts=call_transcripts)
    return insights

# --------------------- Streamlit UI ---------------------
st.title("📝 Earnings Call Insights")
st.write("Obtén un resumen detallado y analítico de la earnings call de cualquier compañía.")

ticker = st.text_input("Ticker de la compañía", value="NVDA")
quarter = st.text_input("Trimestre (ej. 2026Q2)", value="2026Q2")

if st.button("Obtener Insights"):
    if ticker and quarter:
        with st.spinner("Generando insights..."):
            insights = get_call_transcripts(ticker, quarter)
            if insights:
                st.subheader("Resumen Ejecutivo")
                st.write(insights.get("summary"))
                
                st.subheader("Sentimiento General")
                st.write(insights.get("sentiment"))
                
                st.subheader("Temas Clave")
                st.write(insights.get("key_topics"))
                
                st.subheader("Proyecciones / Guidance")
                st.write(insights.get("guidance"))
                
                st.subheader("Cifras Relevantes")
                st.write(insights.get("numeric_highlights"))
                
                st.subheader("Riesgos Detectados")
                st.write(insights.get("risks"))
                
                st.subheader("Preguntas de Analistas")
                st.write(insights.get("analyst_questions"))
                
                st.subheader("Tesis Alcistas")
                st.write(insights.get("bullish_points"))
                
                st.subheader("Tesis Bajistas")
                st.write(insights.get("bearish_points"))
                
                st.subheader("Señales de Alerta")
                st.write(insights.get("red_flags"))
    else:
        st.warning("Por favor ingresa un ticker y un trimestre.")