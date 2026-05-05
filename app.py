"""Dashboard Streamlit del simulador de torneo.

La app es **solo capa de presentación**: toda la lógica vive en ``src/``.
Aquí se hace UI, validación de inputs y orquestación.

Ejecución:
    streamlit run app.py
"""
from typing import Dict, Tuple
import pandas as pd
import streamlit as st

from src.probabilities import ProbabilityMatrix, DEFAULT_PROBABILITIES
from src.monte_carlo import run_simulation
from src.metrics import per_team_metrics, per_level_metrics, compare_scenarios
from src import plots


# --------------------------------------------------------------------------- #
# Configuración de página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Simulador de Torneo (32 equipos · 8 grupos)",
    layout="wide",
    page_icon=None,
)

st.title("Simulador de Torneo")
st.caption(
    "32 equipos · 8 grupos de 4 · 4 niveles · Monte Carlo. "
    "Compara cómo cambian las probabilidades según el sorteo."
)


# --------------------------------------------------------------------------- #
# Estado de probabilidades
# --------------------------------------------------------------------------- #
def _init_probs_state() -> None:
    if "probs" not in st.session_state:
        st.session_state.probs = {
            k: list(v) for k, v in DEFAULT_PROBABILITIES.items()
        }


def _reset_probs() -> None:
    st.session_state.probs = {k: list(v) for k, v in DEFAULT_PROBABILITIES.items()}


_init_probs_state()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Configuración")

    n_sims = st.number_input(
        "Número de simulaciones",
        min_value=100, max_value=100_000, value=2000, step=500,
        help="Más simulaciones = menor varianza, mayor tiempo de cálculo.",
    )
    seed = st.number_input("Semilla aleatoria", min_value=0, value=42)

    mode = st.selectbox(
        "Modo de análisis",
        ["Solo fase de grupos", "Fase de grupos + Fase final"],
    )

    draw_choice = st.selectbox(
        "Sorteo a analizar",
        ["Comparar ambos", "Solo balanceado", "Solo aleatorio"],
    )

    st.divider()
    st.header("Probabilidades por niveles")
    use_defaults = st.checkbox("Usar valores por defecto", value=True,
                               on_change=_reset_probs)

    if not use_defaults:
        st.caption("Ajusta cada par. Si la suma no da 1, se mostrará un aviso.")
        for (i, j) in DEFAULT_PROBABILITIES.keys():
            with st.expander(f"Nivel {i} vs Nivel {j}"):
                w = st.slider(
                    f"P(victoria nivel {i})", 0.0, 1.0,
                    value=float(st.session_state.probs[(i, j)][0]),
                    step=0.01, key=f"w_{i}_{j}",
                )
                d = st.slider(
                    f"P(empate)", 0.0, 1.0,
                    value=float(st.session_state.probs[(i, j)][1]),
                    step=0.01, key=f"d_{i}_{j}",
                )
                l = round(1.0 - w - d, 4)
                if l < -1e-6:
                    st.error(f"P(derrota nivel {i}) = {l:.3f} (negativa). "
                             "Reduce victoria o empate.")
                else:
                    st.info(f"P(derrota nivel {i}) = {max(l, 0.0):.3f} (calculada)")
                st.session_state.probs[(i, j)] = [w, d, max(l, 0.0)]

    st.divider()
    run_button = st.button("Ejecutar simulación", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Construir la matriz y mostrarla
# --------------------------------------------------------------------------- #
try:
    matrix = ProbabilityMatrix.from_dict(
        {k: tuple(v) for k, v in st.session_state.probs.items()}
    )
except ValueError as e:
    st.error(f"Configuración de probabilidades inválida: {e}")
    st.stop()

with st.expander("Matriz de probabilidades en uso", expanded=False):
    st.dataframe(matrix.to_dataframe(), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Ejecución (cacheada)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _run_cached(n_sims: int, draw_type: str, include_knockout: bool,
                seed: int, probs_tuple: Tuple) -> pd.DataFrame:
    """Wrapper cacheado. La clave incluye las probabilidades para invalidar
    correctamente cuando el usuario las cambia."""
    matrix = ProbabilityMatrix.from_dict(dict(probs_tuple))
    return run_simulation(
        n_sims=int(n_sims), draw_type=draw_type,
        matrix=matrix, include_knockout=include_knockout, seed=int(seed),
    )


def _scenarios_to_run(choice: str):
    if choice == "Comparar ambos":
        return [("balanced", "Balanceado"), ("random", "Aleatorio")]
    if choice == "Solo balanceado":
        return [("balanced", "Balanceado")]
    return [("random", "Aleatorio")]


include_knockout = (mode == "Fase de grupos + Fase final")
scenarios = _scenarios_to_run(draw_choice)


if run_button:
    probs_tuple = tuple(sorted(
        (k, tuple(v)) for k, v in st.session_state.probs.items()
    ))
    results: Dict[str, Dict] = {}
    progress = st.progress(0.0, text="Iniciando simulación…")
    for idx, (draw_type, label) in enumerate(scenarios):
        progress.progress((idx + 0.1) / len(scenarios),
                          text=f"Simulando escenario «{label}»…")
        df = _run_cached(int(n_sims), draw_type, include_knockout,
                         int(seed), probs_tuple)
        tm = per_team_metrics(df, include_knockout=include_knockout)
        lm = per_level_metrics(tm)
        results[label] = {"raw": df, "team": tm, "level": lm}
        progress.progress((idx + 1) / len(scenarios),
                          text=f"Escenario «{label}» listo")
    progress.empty()
    st.session_state.results = results
    st.session_state.include_knockout = include_knockout
    st.session_state.last_scenarios = [lbl for _, lbl in scenarios]


# --------------------------------------------------------------------------- #
# Visualización
# --------------------------------------------------------------------------- #
if "results" not in st.session_state:
    st.info("Configura los parámetros en la barra lateral y pulsa «Ejecutar simulación».")
    st.stop()

results = st.session_state.results
include_knockout = st.session_state.include_knockout
labels = st.session_state.last_scenarios

st.header("Resultados")

# --- Vista por escenario ---
tab_labels = labels + (["Comparación"] if len(labels) > 1 else [])
tabs = st.tabs(tab_labels)

for i, label in enumerate(labels):
    data = results[label]
    with tabs[i]:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader(f"Métricas por nivel — {label}")
            st.dataframe(data["level"].round(4), use_container_width=True, hide_index=True)

        with c2:
            st.subheader(f"Métricas por equipo — {label}")
            st.dataframe(data["team"].round(4), use_container_width=True, hide_index=True)

        st.subheader("Probabilidad de clasificación por equipo")
        st.plotly_chart(
            plots.bar_metric_by_team(data["team"], "prob_qualified",
                                     "P(clasificación)", label),
            use_container_width=True,
        )

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                plots.bar_metric_by_team(data["team"], "prob_at_least_one_win",
                                         "P(>=1 victoria)", label),
                use_container_width=True,
            )
        with c4:
            st.plotly_chart(
                plots.bar_metric_by_team(data["team"], "prob_win_and_draw",
                                         "P(>=1V y >=1E)", label),
                use_container_width=True,
            )

        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(
                plots.boxplot_position_by_level(data["raw"]),
                use_container_width=True,
            )
        with c6:
            st.plotly_chart(
                plots.histogram_points_by_level(data["raw"]),
                use_container_width=True,
            )

        if include_knockout:
            st.subheader("Probabilidad de alcanzar cada ronda — por nivel")
            st.plotly_chart(plots.funnel_ko_by_level(data["team"]),
                            use_container_width=True)


# --- Comparación ---
if len(labels) > 1:
    with tabs[-1]:
        a, b = labels[0], labels[1]
        level_a = results[a]["level"]
        level_b = results[b]["level"]

        st.subheader(f"Tabla comparativa por nivel — {a} vs {b}")
        comparison = compare_scenarios(level_a, level_b, label_a=a, label_b=b)
        st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)

        for metric, title in [
            ("prob_qualified",        "P(clasificación) por nivel"),
            ("prob_at_least_one_win", "P(>=1 victoria) por nivel"),
            ("prob_win_and_draw",     "P(>=1V y >=1E) por nivel"),
        ]:
            st.plotly_chart(
                plots.comparison_bar_by_level(level_a, level_b, metric, title, a, b),
                use_container_width=True,
            )

        if include_knockout:
            for metric, title in [
                ("prob_reach_quarterfinals", "P(llegar a cuartos) por nivel"),
                ("prob_reach_semifinals",    "P(llegar a semifinales) por nivel"),
                ("prob_reach_final",         "P(llegar a la final) por nivel"),
                ("prob_reach_champion",      "P(ser campeón) por nivel"),
            ]:
                if f"{metric}_mean" in level_a.columns:
                    st.plotly_chart(
                        plots.comparison_bar_by_level(level_a, level_b, metric, title, a, b),
                        use_container_width=True,
                    )
