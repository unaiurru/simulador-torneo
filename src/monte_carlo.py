"""Orquestador Monte Carlo.

Ejecuta N simulaciones independientes y devuelve un DataFrame "long" con una
fila por (simulación, equipo). Esto permite calcular cualquier métrica
posterior con pandas sin guardar intermedios.

Reproducibilidad: la función crea un ``Generator`` maestro a partir de
``seed`` y deriva un nuevo ``Generator`` por simulación. Esto garantiza que:
    - Dos llamadas con la misma semilla producen DataFrames idénticos.
    - Las simulaciones son independientes entre sí.
"""
from typing import List, Optional
import numpy as np
import pandas as pd

from .teams import Team, create_teams
from .probabilities import ProbabilityMatrix
from .group_draw import make_draw
from .group_stage import simulate_all_groups
from .knockout_stage import build_bracket, simulate_knockout, ROUND_RANK


def run_simulation(n_sims: int,
                   draw_type: str,
                   matrix: ProbabilityMatrix,
                   include_knockout: bool = False,
                   seed: int = 42,
                   teams: Optional[List[Team]] = None) -> pd.DataFrame:
    """Corre ``n_sims`` simulaciones y devuelve el DataFrame de resultados.

    Args:
        n_sims: número de torneos a simular.
        draw_type: ``"balanced"`` o ``"random"``.
        matrix: matriz de probabilidades a usar.
        include_knockout: si True, simula también la fase eliminatoria.
        seed: semilla maestra para reproducibilidad.
        teams: lista de equipos (por defecto los 32 estándar).

    Returns:
        DataFrame con columnas:
            simulation, team_id, team_name, level, group, position,
            points, wins, draws, losses, qualified, at_least_one_win,
            win_and_draw, [knockout_round, ko_rank]
    """
    if n_sims < 1:
        raise ValueError("n_sims debe ser >= 1")
    if teams is None:
        teams = create_teams()

    master_rng = np.random.default_rng(seed)
    rows = []

    for sim_idx in range(n_sims):
        sim_seed = int(master_rng.integers(0, 2**31 - 1))
        sim_rng = np.random.default_rng(sim_seed)

        groups = make_draw(draw_type, teams, sim_rng)
        standings = simulate_all_groups(groups, matrix, sim_rng)

        ko_rounds = {}
        if include_knockout:
            bracket = build_bracket(standings, sim_rng)
            ko_rounds = simulate_knockout(bracket, matrix, sim_rng)

        for g_idx, group_standings in enumerate(standings):
            for pos, stats in enumerate(group_standings, start=1):
                team = stats.team
                row = {
                    "simulation": sim_idx,
                    "team_id": team.id,
                    "team_name": team.name,
                    "level": team.level,
                    "group": g_idx,
                    "position": pos,
                    "points": stats.points,
                    "wins": stats.wins,
                    "draws": stats.draws,
                    "losses": stats.losses,
                    "qualified": pos <= 2,
                    "at_least_one_win": stats.wins >= 1,
                    "win_and_draw": (stats.wins >= 1) and (stats.draws >= 1),
                }
                if include_knockout:
                    round_name = ko_rounds.get(team.id, "did_not_qualify")
                    row["knockout_round"] = round_name
                    row["ko_rank"] = ROUND_RANK[round_name]
                rows.append(row)

    return pd.DataFrame(rows)
