"""Gráficos Plotly de la sección teórica.

Cada función devuelve un `plotly.graph_objects.Figure` listo para `st.plotly_chart`.

Convenciones de visualización (instrucciones del proyecto):
- Mismo color por sorteo (aleatorio = azul, balanceado = naranja).
- Misma escala vertical [0, 1] para todas las curvas de probabilidad.
- Títulos, ejes y leyendas con unidades y significado explícito.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src import theory


COLOR_ALE = "#1f77b4"   # azul
COLOR_BAL = "#ff7f0e"   # naranja
COLOR_INFLEX = "#2ca02c"
COLOR_CURV = "#d62728"


# --------------------------------------------------------------------------- #
# 1. Sigmoide con punto de inflexión y zonas de curvatura
# --------------------------------------------------------------------------- #
def plot_sigmoid_with_inflection(steepness: float, x_range: float = 3.0) -> go.Figure:
    """σ(x) con su punto de inflexión y zonas cóncava/convexa sombreadas."""
    x = np.linspace(-x_range, x_range, 400)
    y = theory.sigmoid(x, k=steepness)

    fig = go.Figure()
    # Zona convexa (σ < 1/2) – favorece a los débiles con varianza
    mask_lo = x <= 0
    fig.add_trace(go.Scatter(
        x=x[mask_lo], y=y[mask_lo], mode="lines",
        name="σ < ½  (convexa, favorece a equipos débiles)",
        line=dict(color=COLOR_ALE, width=3),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.10)",
        hovertemplate="x=%{x:.2f}<br>σ=%{y:.3f}<extra></extra>",
    ))
    # Zona cóncava (σ > 1/2) – perjudica a los fuertes con varianza
    mask_hi = x >= 0
    fig.add_trace(go.Scatter(
        x=x[mask_hi], y=y[mask_hi], mode="lines",
        name="σ > ½  (cóncava, perjudica a equipos fuertes)",
        line=dict(color=COLOR_CURV, width=3),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.10)",
        hovertemplate="x=%{x:.2f}<br>σ=%{y:.3f}<extra></extra>",
    ))
    # Línea P = 1/2
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="P = ½", annotation_position="bottom right")
    # Punto de inflexión
    fig.add_trace(go.Scatter(
        x=[0], y=[0.5], mode="markers+text",
        marker=dict(size=14, color=COLOR_INFLEX, symbol="x"),
        text=["cambio de curvatura"], textposition="top center",
        showlegend=False,
    ))

    titulo = (
        f"Sigmoide σ(s − μ) con pendiente k = {steepness:.2f}"
        if steepness > 0 else
        "Pendiente k = 0  ⇒  σ ≡ ½  (recta horizontal)"
    )
    fig.update_layout(
        title=titulo,
        xaxis_title="x = s_i − μ_rival   (ventaja de fuerza)",
        yaxis_title="P(clasificar)",
        yaxis=dict(range=[-0.02, 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        height=420,
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig


# --------------------------------------------------------------------------- #
# 2. Curvas P_ale y P_bal por rango
# --------------------------------------------------------------------------- #
def plot_prob_curves(
    strengths: np.ndarray,
    steepness: float,
    m: int = 4,
) -> go.Figure:
    """Compara las dos curvas teóricas i ↦ P^ale_i  e  i ↦ P^bal_i."""
    p_ale = theory.qualification_curve(strengths, steepness, "random", m=m)
    p_bal = theory.qualification_curve(strengths, steepness, "balanced", m=m)
    ranks = np.arange(1, len(strengths) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ranks, y=p_ale, mode="lines+markers",
        name="P^ale  (sorteo aleatorio)",
        line=dict(color=COLOR_ALE, width=3),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=ranks, y=p_bal, mode="lines+markers",
        name="P^bal  (sorteo balanceado)",
        line=dict(color=COLOR_BAL, width=3),
        marker=dict(size=7),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="promedio teórico = ½",
                  annotation_position="top right")

    # Cruces entre las curvas
    crossings = theory.find_crossings(p_bal, p_ale)
    for c in crossings:
        # Punto medio entre i y i+1 visualmente
        fig.add_vline(
            x=ranks[c] + 0.5, line_dash="dot", line_color=COLOR_INFLEX,
            annotation_text="cruce", annotation_position="top",
        )

    avg_ale = theory.average_probability(p_ale)
    avg_bal = theory.average_probability(p_bal)
    fig.update_layout(
        title=(
            f"Curvas teóricas P_i por rango   ·   "
            f"promedio_ale = {avg_ale:.3f}   ·   promedio_bal = {avg_bal:.3f}"
        ),
        xaxis_title="Rango del equipo (1 = más fuerte, n = más débil)",
        yaxis_title="P(clasificar)",
        yaxis=dict(range=[-0.02, 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        height=460,
        margin=dict(l=40, r=20, t=70, b=80),
    )
    return fig


# --------------------------------------------------------------------------- #
# 3. Curvatura: σ y σ'' en el mismo plot
# --------------------------------------------------------------------------- #
def plot_curvature_demo(steepness: float, x_range: float = 3.0) -> go.Figure:
    """σ y σ'' (curvatura) en doble eje, para ver dónde cambia el signo."""
    x = np.linspace(-x_range, x_range, 400)
    y = theory.sigmoid(x, k=steepness)
    y2 = theory.sigmoid_second_derivative(x, k=steepness)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name="σ(x)",
        line=dict(color=COLOR_ALE, width=3),
        hovertemplate="x=%{x:.2f}<br>σ=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y2, mode="lines", name="σ''(x) — curvatura",
        line=dict(color=COLOR_CURV, width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="x=%{x:.2f}<br>σ''=%{y:.4f}<extra></extra>",
    ))
    # Refs visuales
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.0, line_dash="dot", line_color=COLOR_INFLEX,
                  annotation_text="σ''(0)=0", annotation_position="top right")

    fig.update_layout(
        title="σ y su segunda derivada — σ'' cambia de signo cuando σ = ½",
        xaxis_title="x = s − μ",
        yaxis=dict(title="σ(x)", range=[-0.02, 1.02]),
        yaxis2=dict(
            title="σ''(x)", overlaying="y", side="right",
            showgrid=False, zeroline=True, zerolinecolor="gray",
        ),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(l=40, r=60, t=60, b=80),
    )
    return fig
