# capital_eye
Herramienta financiera para conocer el panorama general del mercado financiero estadounidense

## Instalacion
1. Clona este repositorio y sitúate en su carpeta raíz.
2. Crea un entorno virtual con Python 3.13 o superior (por ejemplo python -m venv .venv) y actívalo.
3. Instala uv si aún no lo tienes (pip install uv) y luego ejecuta uv install para resolver dependencias usando pyproject.toml.
4. Copia .env.example a .env y rellena OPENAI_API_KEY y ALPHAVANTAGE_API_KEY con tus claves secretas.
5. Ejecuta la aplicación con uv run streamlit run interfaz.py y abre el navegador cuando Streamlit indique la URL local.

## Descarga dataset 
En caso de que el usuario desee el dataset en tiempo real de panorama_df, favor de correr el codigo main_05.py para 
poder obtenerlo, mientras se puede usar el dataset panorama_df que se encuentra en el repositorio

## Arbol de dependencias

```text
capital-eye 
├── beautifulsoup4 v4.14.3
│   ├── soupsieve v2.8.3
│   └── typing-extensions v4.15.0
├── matplotlib v3.10.8
│   ├── contourpy v1.3.3
│   │   └── numpy v2.4.4
│   ├── cycler v0.12.1
│   ├── fonttools v4.62.1
│   ├── kiwisolver v1.5.0
│   ├── numpy v2.4.4
│   ├── packaging v26.0
│   ├── pillow v12.2.0
│   ├── pyparsing v3.3.2
│   └── python-dateutil v2.9.0.post0
│       └── six v1.17.0
├── nltk v3.9.4
│   ├── click v8.3.1
│   ├── joblib v1.5.3
│   ├── regex v2026.4.4
│   └── tqdm v4.67.3
├── openai v2.30.0
│   ├── anyio v4.13.0
│   │   └── idna v3.11
│   ├── distro v1.9.0
│   ├── httpx v0.28.1
│   │   ├── anyio v4.13.0 (*)
│   │   ├── certifi v2026.2.25
│   │   ├── httpcore v1.0.9
│   │   │   ├── certifi v2026.2.25
│   │   │   └── h11 v0.16.0
│   │   └── idna v3.11
│   ├── jiter v0.13.0
│   ├── pydantic v2.12.5
│   │   ├── annotated-types v0.7.0
│   │   ├── pydantic-core v2.41.5
│   │   │   └── typing-extensions v4.15.0
│   │   ├── typing-extensions v4.15.0
│   │   └── typing-inspection v0.4.2
│   │       └── typing-extensions v4.15.0
│   ├── sniffio v1.3.1
│   ├── tqdm v4.67.3
│   └── typing-extensions v4.15.0
├── plotly v6.6.0
│   ├── narwhals v2.18.1
│   └── packaging v26.0
├── python-dotenv v1.2.2
├── requests v2.33.1
│   ├── certifi v2026.2.25
│   ├── charset-normalizer v3.4.7
│   ├── idna v3.11
│   └── urllib3 v2.6.3
├── scikit-learn v1.8.0
│   ├── joblib v1.5.3
│   ├── numpy v2.4.4
│   ├── scipy v1.17.1
│   │   └── numpy v2.4.4
│   └── threadpoolctl v3.6.0
├── seaborn v0.13.2
│   ├── matplotlib v3.10.8 (*)
│   ├── numpy v2.4.4
│   └── pandas v3.0.2
│       ├── numpy v2.4.4
│       └── python-dateutil v2.9.0.post0 (*)
├── streamlit v1.56.0
│   ├── altair v6.0.0
│   │   ├── jinja2 v3.1.6
│   │   │   └── markupsafe v3.0.3
│   │   ├── jsonschema v4.26.0
│   │   │   ├── attrs v26.1.0
│   │   │   ├── jsonschema-specifications v2025.9.1
│   │   │   │   └── referencing v0.37.0
│   │   │   │       ├── attrs v26.1.0
│   │   │   │       └── rpds-py v0.30.0
│   │   │   ├── referencing v0.37.0 (*)
│   │   │   └── rpds-py v0.30.0
│   │   ├── narwhals v2.18.1
│   │   ├── packaging v26.0
│   │   └── typing-extensions v4.15.0
│   ├── blinker v1.9.0
│   ├── cachetools v7.0.5
│   ├── click v8.3.1
│   ├── gitpython v3.1.46
│   │   └── gitdb v4.0.12
│   │       └── smmap v5.0.3
│   ├── numpy v2.4.4
│   ├── packaging v26.0
│   ├── pandas v3.0.2 (*)
│   ├── pillow v12.2.0
│   ├── protobuf v7.34.1
│   ├── pyarrow v23.0.1
│   ├── pydeck v0.9.1
│   │   ├── jinja2 v3.1.6 (*)
│   │   └── numpy v2.4.4
│   ├── requests v2.33.1 (*)
│   ├── tenacity v9.1.4
│   ├── toml v0.10.2
│   ├── tornado v6.5.5
│   └── typing-extensions v4.15.0
├── uvicorn v0.43.0
│   ├── click v8.3.1
│   └── h11 v0.16.0
└── watchdog v6.0.0
```
