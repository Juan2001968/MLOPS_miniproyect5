from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update
from dash import dash_table  # ✅ DataTable va aquí en Dash moderno
import plotly.express as px


# -----------------------------------------------------------
# Utilidad: cargar algún dataset disponible del proyecto
# -----------------------------------------------------------
def _load_any_data() -> pd.DataFrame:
    """
    Busca un parquet razonable para el dashboard. Devuelve un DataFrame pequeño.
    Prioriza X_test procesado; si no existe, intenta otros. Si nada existe,
    genera un dataset sintético para que el dashboard siempre levante.
    """
    candidates = [
        Path("data/processed/X_test.parquet"),
        Path("data/interim/X_test.parquet"),
        Path("data/processed/telco_churn.parquet"),
        Path("docs/data/processed/X_test.parquet"),  # por si el libro copia data a docs
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            # Evita dataframes gigantes en la tabla
            if len(df) > 1000:
                df = df.sample(1000, random_state=42)
            return df.reset_index(drop=True)

    # Fallback: dataset sintético para no romper el dashboard
    import numpy as np

    n = 300
    df = pd.DataFrame(
        {
            "tenure": np.random.randint(1, 72, size=n),
            "MonthlyCharges": np.random.uniform(15, 130, size=n).round(2),
            "TotalCharges": np.random.uniform(20, 8000, size=n).round(2),
            "InternetService_Fiber": np.random.randint(0, 2, size=n),
            "SeniorCitizen": np.random.randint(0, 2, size=n),
        }
    )
    return df


# -----------------------------------------------------------
# Fábrica de la app Dash
# -----------------------------------------------------------
def create_dash_app() -> Dash:
    """
    Crea y devuelve una instancia de Dash lista para montar en FastAPI
    (vía WSGIMiddleware en `src/app/main.py`). NO usa `NO_UPDATE` y
    utiliza `dash_table.DataTable` correctamente.
    """
    df = _load_any_data()
    numeric_cols = df.select_dtypes("number").columns.tolist()
    all_cols = df.columns.tolist()

    # Valores por defecto para gráfica
    default_x = numeric_cols[0] if numeric_cols else all_cols[0]
    default_y = numeric_cols[1] if len(numeric_cols) > 1 else None

    # requests_pathname_prefix es clave cuando montas el dashboard en /dash
    dash_app = Dash(
        __name__,
        requests_pathname_prefix="/dash/",
        title="Churn Dashboard",
        suppress_callback_exceptions=True,
    )

    dash_app.layout = html.Div(
        className="container",
        children=[
            html.H2("Dashboard — Churn (Demo)"),
            html.Div(
                style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Eje X"),
                            dcc.Dropdown(
                                id="x-col",
                                options=[{"label": c, "value": c} for c in all_cols],
                                value=default_x,
                                clearable=False,
                            ),
                        ],
                        style={"minWidth": "220px"},
                    ),
                    html.Div(
                        children=[
                            html.Label("Eje Y (opcional)"),
                            dcc.Dropdown(
                                id="y-col",
                                options=[{"label": c, "value": c} for c in numeric_cols],
                                value=default_y,
                                clearable=True,
                            ),
                        ],
                        style={"minWidth": "220px"},
                    ),
                    html.Div(
                        children=[
                            html.Label("Tipo de gráfica"),
                            dcc.Dropdown(
                                id="chart-type",
                                options=[
                                    {"label": "Histograma", "value": "hist"},
                                    {"label": "Dispersión", "value": "scatter"},
                                    {"label": "Boxplot", "value": "box"},
                                ],
                                value="hist",
                                clearable=False,
                            ),
                        ],
                        style={"minWidth": "220px"},
                    ),
                ],
            ),
            html.Br(),
            dcc.Graph(id="main-fig"),
            html.Hr(),
            html.H4("Vista de datos (hasta 1.000 filas)"),
            dash_table.DataTable(
                id="data-table",
                columns=[{"name": c, "id": c} for c in df.columns],
                data=df.to_dict("records"),
                page_size=12,
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "sans-serif", "fontSize": 13},
                filter_action="native",
                sort_action="native",
            ),
            # Memoria para no recalcular el DF en cada callback
            dcc.Store(id="mem-data", data=df.to_json(date_format="iso", orient="split")),
        ],
    )

    # -------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------
    @dash_app.callback(
        Output("main-fig", "figure"),
        Input("x-col", "value"),
        Input("y-col", "value"),
        Input("chart-type", "value"),
        State("mem-data", "data"),
        prevent_initial_call=False,
    )
    def update_figure(x_col, y_col, chart_type, mem_json):
        # Recuperar df desde memoria
        try:
            _df = pd.read_json(mem_json, orient="split")
        except Exception:
            _df = df  # fallback

        if not x_col:
            return no_update

        if chart_type == "hist":
            fig = px.histogram(_df, x=x_col, nbins=40, title=f"Histograma — {x_col}")
        elif chart_type == "scatter":
            if y_col is None:
                return no_update
            fig = px.scatter(_df, x=x_col, y=y_col, opacity=0.7, title=f"Dispersión — {x_col} vs {y_col}")
        elif chart_type == "box":
            fig = px.box(_df, x=x_col, title=f"Boxplot — {x_col}")
        else:
            fig = px.histogram(_df, x=x_col, nbins=40)

        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        return fig

    return dash_app

