"""Cálculo de métricas a partir del DataFrame de simulaciones.

Tres niveles de agregación:
    1. ``per_team_metrics``      — una fila por equipo, métricas medias.
    2. ``per_level_metrics``     — una fila por nivel, con media y std
                                   *entre equipos* del mismo nivel.
    3. ``compare_scenarios``     — diferencias absolutas y relativas entre
                                   dos escenarios al nivel de "nivel".

Distinción importante: la **probabilidad individual** es la media a lo largo
de las simulaciones para un equipo concreto. La **media por nivel** es la
media de esas probabilidades entre los equipos del mismo nivel. La **std por
nivel** mide la dispersión de probabilidades entre equipos del mismo nivel,
que es exactamente la "varianza entre equipos" que pediste.
"""
from typing import List
import numpy as np
import pandas as pd

from .knockout_stage import ROUND_RANK


# --------------------------------------------------------------------------- #
# Métricas por equipo
# --------------------------------------------------------------------------- #
def per_team_metrics(df: pd.DataFrame,
                     include_knockout: bool = False) -> pd.DataFrame:
    """Agrega el DataFrame "long" a una fila por equipo."""
    grouped = df.groupby(["team_id", "team_name", "level"])

    metrics = grouped.agg(
        prob_qualified=("qualified", "mean"),
        prob_at_least_one_win=("at_least_one_win", "mean"),
        prob_win_and_draw=("win_and_draw", "mean"),
        avg_points=("points", "mean"),
        std_points=("points", "std"),
        avg_position=("position", "mean"),
        std_position=("position", "std"),
    ).reset_index()

    if include_knockout and "knockout_round" in df.columns:
        ko = _compute_ko_probabilities(df)
        metrics = metrics.merge(ko, on=["team_id", "team_name", "level"], how="left")

    return metrics


def _compute_ko_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de **alcanzar o pasar** cada ronda eliminatoria."""
    rounds = ["round_of_16", "quarterfinals", "semifinals", "final", "champion"]

    out_rows = []
    grouped = df.groupby(["team_id", "team_name", "level"])
    for (team_id, team_name, level), g in grouped:
        row = {"team_id": team_id, "team_name": team_name, "level": level}
        for r in rounds:
            row[f"prob_reach_{r}"] = float((g["ko_rank"] >= ROUND_RANK[r]).mean())
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# Métricas por nivel
# --------------------------------------------------------------------------- #
def per_level_metrics(team_metrics: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por nivel: media y std *entre equipos del mismo nivel*."""
    metric_cols: List[str] = [
        c for c in team_metrics.columns
        if c.startswith(("prob_", "avg_"))
    ]

    rows = []
    for level, g in team_metrics.groupby("level"):
        row = {"level": int(level), "n_teams": int(len(g))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("level").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Comparativa entre escenarios
# --------------------------------------------------------------------------- #
def compare_scenarios(level_metrics_a: pd.DataFrame,
                      level_metrics_b: pd.DataFrame,
                      label_a: str = "A",
                      label_b: str = "B") -> pd.DataFrame:
    """Compara dos escenarios a nivel de 'nivel'.

    Calcula diferencia absoluta (A - B) y relativa ((A - B) / B) para cada
    métrica '_mean'. Las std permanecen para cada escenario por separado.
    """
    merged = level_metrics_a.merge(
        level_metrics_b, on="level", suffixes=(f"_{label_a}", f"_{label_b}")
    )

    bases = sorted({c[:-5] for c in level_metrics_a.columns if c.endswith("_mean")})
    for base in bases:
        col_a = f"{base}_mean_{label_a}"
        col_b = f"{base}_mean_{label_b}"
        if col_a in merged.columns and col_b in merged.columns:
            merged[f"{base}_diff_abs"] = merged[col_a] - merged[col_b]
            with np.errstate(divide="ignore", invalid="ignore"):
                merged[f"{base}_diff_rel"] = np.where(
                    np.abs(merged[col_b]) > 1e-12,
                    (merged[col_a] - merged[col_b]) / merged[col_b],
                    np.nan,
                )
    return merged
