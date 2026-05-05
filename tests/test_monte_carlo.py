"""Tests de Monte Carlo (reproducibilidad y sanidad estadística)."""
import numpy as np
import pandas as pd
from src.monte_carlo import run_simulation
from src.probabilities import ProbabilityMatrix


def test_simulation_reproducibility_balanced():
    matrix = ProbabilityMatrix.from_default()
    df1 = run_simulation(n_sims=20, draw_type="balanced", matrix=matrix, seed=123)
    df2 = run_simulation(n_sims=20, draw_type="balanced", matrix=matrix, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_simulation_reproducibility_random():
    matrix = ProbabilityMatrix.from_default()
    df1 = run_simulation(n_sims=20, draw_type="random", matrix=matrix, seed=99)
    df2 = run_simulation(n_sims=20, draw_type="random", matrix=matrix, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_two_qualified_per_group_per_sim():
    matrix = ProbabilityMatrix.from_default()
    df = run_simulation(n_sims=10, draw_type="random", matrix=matrix, seed=42)
    per_sim_group = df.groupby(["simulation", "group"])["qualified"].sum()
    assert (per_sim_group == 2).all()


def test_team_appears_once_per_sim():
    matrix = ProbabilityMatrix.from_default()
    df = run_simulation(n_sims=15, draw_type="balanced", matrix=matrix, seed=7)
    counts = df.groupby(["simulation", "team_id"]).size()
    assert (counts == 1).all()


def test_qualification_rate_orders_by_level_in_balanced():
    """En sorteo balanceado los equipos de nivel 1 deberían clasificar más
    que los de nivel 4, y muy claramente con 1500 simulaciones."""
    matrix = ProbabilityMatrix.from_default()
    df = run_simulation(n_sims=1500, draw_type="balanced", matrix=matrix, seed=42)
    by_level = df.groupby("level")["qualified"].mean()
    assert by_level[1] > by_level[2] > by_level[3] > by_level[4]
    assert by_level[1] > 0.7
    assert by_level[4] < 0.3


def test_same_level_teams_have_similar_qualification_rate():
    """Equipos del mismo nivel deben tener tasas de clasificación parecidas
    (con suficientes simulaciones). Usamos un test laxo: rango < 0.1."""
    matrix = ProbabilityMatrix.from_default()
    df = run_simulation(n_sims=3000, draw_type="balanced", matrix=matrix, seed=11)
    rates = df.groupby(["level", "team_id"])["qualified"].mean()
    for level in (1, 2, 3, 4):
        level_rates = rates.loc[level]
        assert level_rates.max() - level_rates.min() < 0.1, \
            f"Dispersión sospechosa en nivel {level}: {level_rates.to_dict()}"


def test_knockout_columns_present_when_enabled():
    matrix = ProbabilityMatrix.from_default()
    df = run_simulation(n_sims=5, draw_type="balanced", matrix=matrix,
                        include_knockout=True, seed=42)
    assert "knockout_round" in df.columns
    assert "ko_rank" in df.columns
    # No clasificados deben tener round = 'did_not_qualify'
    not_qualified = df[~df["qualified"]]
    assert (not_qualified["knockout_round"] == "did_not_qualify").all()


def test_knockout_exactly_one_champion_per_sim():
    matrix = ProbabilityMatrix.from_default()
    df = run_simulation(n_sims=10, draw_type="random", matrix=matrix,
                        include_knockout=True, seed=42)
    champs_per_sim = df.groupby("simulation")["knockout_round"].apply(
        lambda s: (s == "champion").sum()
    )
    assert (champs_per_sim == 1).all()


def test_balanced_vs_random_lower_variance_for_top_level():
    """En sorteo balanceado los equipos de nivel 1 deberían tener una tasa
    de clasificación más alta en media (porque siempre les tocan rivales
    de niveles más bajos) que en sorteo aleatorio.
    """
    matrix = ProbabilityMatrix.from_default()
    df_b = run_simulation(n_sims=2000, draw_type="balanced", matrix=matrix, seed=1)
    df_r = run_simulation(n_sims=2000, draw_type="random", matrix=matrix, seed=1)
    rate_b = df_b[df_b["level"] == 1]["qualified"].mean()
    rate_r = df_r[df_r["level"] == 1]["qualified"].mean()
    assert rate_b > rate_r
