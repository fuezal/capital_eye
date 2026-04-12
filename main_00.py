# ==========================================================================================
# Codigo individual: LOGO Capitaleye
# ==========================================================================================

# LIBRERIAS
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
st.write("Aquí va tu contenido del dashboard...")