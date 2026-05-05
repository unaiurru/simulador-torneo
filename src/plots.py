"""Visualizaciones Plotly.

Funciones puras: reciben DataFrames y devuelven ``go.Figure``. No tocan
Streamlit ni hacen ``fig.show()``: la capa de presentación se encarga.

Las escalas de probabilidad se fijan a [0, 1] para que las comparaciones
entre escenarios sean justas visualmente.
"""
from typing import Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


LEVEL_COLORS: Dict[int, str] = {
    1: "#1f77b4",  # azul
    2: "#2ca02c",  # verde
    3: "#ff7f0e",  # naranja
    4: "#d62728",  # rojo
}

LEVEL_COLORS_STR: Dict[str, str] = {str(k): v for k, v in LEVEL_COLORS.items()}

ROUND_LABELS = {
    "round_of_16":   "Octavos",
    "quarterfinals": "Cuartos",
    "semifinals":    "Semifinales",
    "final":         "Final",
    "champion":      "Campeón",
}


def _prob_layout(fig: go.Figure, y_title: str) -> go.Figure:
    fig.update_layout(yaxis=dict(range=[0, 1], title=y_title),
                      legend=dict(title="Nivel"))
    return fig


# --------------------------------------------------------------------------- #
# Barras por equipo (resultados detallados)
# --------------------------------------------------------------------------- #
def bar_metric_by_team(team_metrics: pd.DataFrame,
                       metric_col: str,
                       title: str,
                       scenario_label: str = "") -> go.Figure:
    """Barras de una métrica por equipo, coloreadas por nivel."""
    df = team_metrics.sort_values(["level", "team_name"])
    suffix = f" — {scenario_label}" if scenario_label else ""
    fig = px.bar(
        df,
        x="team_name", y=metric_col,
        color=df["level"].astype(str),
        color_discrete_map=LEVEL_COLORS_STR,
        title=f"{title}{suffix}",
        labels={"team_name": "Equipo", metric_col: title, "color": "Nivel"},
    )
    if metric_col.startswith("prob_"):
        _prob_layout(fig, title)
    return fig


# --------------------------------------------------------------------------- #
# Comparativa entre escenarios
# --------------------------------------------------------------------------- #
def comparison_bar_by_level(level_a: pd.DataFrame,
                            level_b: pd.DataFrame,
                            metric: str,
                            title: str,
                            label_a: str = "Balanceado",
                            label_b: str = "Aleatorio") -> go.Figure:
    """Barras agrupadas por nivel para comparar dos escenarios.

    `metric` debe corresponder al prefijo (sin '_mean'), por ejemplo
    ``"prob_qualified"``; la función busca ``"<metric>_mean"``.
    """
    col = f"{metric}_mean"
    err_col = f"{metric}_std"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=label_a,
        x=level_a["level"].astype(str),
        y=level_a[col],
        error_y=dict(type="data", array=level_a.get(err_col, None), visible=err_col in level_a.columns),
        marker_color="#4c78a8",
    ))
    fig.add_trace(go.Bar(
        name=label_b,
        x=level_b["level"].astype(str),
        y=level_b[col],
        error_y=dict(type="data", array=level_b.get(err_col, None), visible=err_col in level_b.columns),
        marker_color="#f58518",
    ))
    fig.update_layout(
        title=title, barmode="group",
        xaxis_title="Nivel", yaxis_title=metric.replace("_", " "),
    )
    if metric.startswith("prob_"):
        fig.update_layout(yaxis=dict(range=[0, 1]))
    return fig


# --------------------------------------------------------------------------- #
# Distribuciones a partir del DataFrame raw
# --------------------------------------------------------------------------- #
def boxplot_position_by_level(df: pd.DataFrame,
                              title: str = "Posición en grupo por nivel") -> go.Figure:
    """Boxplot de posición final en grupo por nivel."""
    fig = px.box(
        df, x=df["level"].astype(str), y="position",
        color=df["level"].astype(str),
        color_discrete_map=LEVEL_COLORS_STR,
        title=title, labels={"x": "Nivel", "position": "Posición (1=mejor)"},
    )
    fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
    return fig


def histogram_points_by_level(df: pd.DataFrame) -> go.Figure:
    """Histograma de puntos en fase de grupos por nivel."""
    fig = px.histogram(
        df, x="points",
        color=df["level"].astype(str),
        color_discrete_map=LEVEL_COLORS_STR,
        barmode="overlay", nbins=10,
        title="Distribución de puntos en fase de grupos por nivel",
        labels={"points": "Puntos", "color": "Nivel"},
    )
    fig.update_layout(legend_title="Nivel")
    return fig


# --------------------------------------------------------------------------- #
# Fase final
# --------------------------------------------------------------------------- #
def funnel_ko_by_level(team_metrics: pd.DataFrame,
                       title: str = "Probabilidad de alcanzar cada ronda por nivel") -> go.Figure:
    """Barras agrupadas con la prob. media (sobre equipos del nivel) de
    alcanzar cada ronda.
    """
    rounds = ["round_of_16", "quarterfinals", "semifinals", "final", "champion"]
    rows = []
    for level, g in team_metrics.groupby("level"):
        for r in rounds:
            col = f"prob_reach_{r}"
            if col in g.columns:
                rows.append({
                    "Nivel": str(level),
                    "Ronda": ROUND_LABELS[r],
                    "Probabilidad": float(g[col].mean()),
                })
    df_long = pd.DataFrame(rows)
    if df_long.empty:
        return go.Figure()
    df_long["Ronda"] = pd.Categorical(
        df_long["Ronda"],
        categories=[ROUND_LABELS[r] for r in rounds],
        ordered=True,
    )
    fig = px.bar(
        df_long.sort_values(["Ronda", "Nivel"]),
        x="Ronda", y="Probabilidad", color="Nivel",
        barmode="group", color_discrete_map=LEVEL_COLORS_STR,
        title=title,
    )
    fig.update_layout(yaxis=dict(range=[0, 1]))
    return fig
