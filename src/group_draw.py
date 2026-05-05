"""Generación de grupos.

Dos modos:
- ``balanced``: solo los equipos del nivel 1 ("los mejores") se reparten
  forzadamente uno por grupo (estilo cabezas de serie). Los 24 equipos
  restantes (niveles 2, 3 y 4) se barajan juntos y se reparten 3 por grupo
  sin restricción de nivel.
- ``random``: los 32 equipos se barajan y se reparten en 8 grupos de 4 sin
  ninguna restricción.

Ambas funciones reciben un ``numpy.random.Generator`` para garantizar
reproducibilidad. Devuelven una lista de 8 grupos (cada uno una lista de Team).
"""
from typing import List
import numpy as np

from .teams import Team, NUM_GROUPS, TEAMS_PER_GROUP


# Nivel que se trata como "cabeza de serie" en el sorteo balanceado.
TOP_SEED_LEVEL = 1


def balanced_draw(teams: List[Team],
                  rng: np.random.Generator) -> List[List[Team]]:
    """Sorteo balanceado por cabezas de serie (solo nivel 1).

    Algoritmo:
        1. Separar los equipos de nivel 1 (cabezas de serie) del resto.
        2. Barajar los nivel 1 y asignar exactamente 1 a cada grupo.
        3. Barajar los demás equipos juntos (sin importar su nivel) y
           rellenar los huecos restantes (3 por grupo).

    Garantías:
        - Cada grupo tiene exactamente 1 equipo de nivel 1.
        - Cada equipo no-nivel-1 puede caer en cualquier grupo con la misma
          probabilidad (1/8), independientemente de su nivel concreto.
        - Equipos del mismo nivel siguen siendo intercambiables.
    """
    by_level: dict = {}
    for t in teams:
        by_level.setdefault(t.level, []).append(t)

    if TOP_SEED_LEVEL not in by_level:
        raise ValueError(
            f"No hay equipos de nivel {TOP_SEED_LEVEL} (cabezas de serie)"
        )
    if len(by_level[TOP_SEED_LEVEL]) != NUM_GROUPS:
        raise ValueError(
            f"Sorteo balanceado requiere exactamente {NUM_GROUPS} equipos de "
            f"nivel {TOP_SEED_LEVEL}, got {len(by_level[TOP_SEED_LEVEL])}"
        )
    if len(teams) != NUM_GROUPS * TEAMS_PER_GROUP:
        raise ValueError(
            f"Sorteo balanceado requiere {NUM_GROUPS * TEAMS_PER_GROUP} equipos "
            f"en total, got {len(teams)}"
        )

    # 1) Cabezas de serie: 1 nivel-1 por grupo (barajados).
    top_seeds = by_level[TOP_SEED_LEVEL].copy()
    rng.shuffle(top_seeds)
    groups: List[List[Team]] = [[t] for t in top_seeds]

    # 2) Resto: niveles 2..N juntos, repartidos aleatoriamente.
    others: List[Team] = []
    for lvl, lst in by_level.items():
        if lvl != TOP_SEED_LEVEL:
            others.extend(lst)
    rng.shuffle(others)

    slots_per_group = TEAMS_PER_GROUP - 1  # 3
    for i, t in enumerate(others):
        groups[i // slots_per_group].append(t)

    return groups


def random_draw(teams: List[Team],
                rng: np.random.Generator) -> List[List[Team]]:
    """Sorteo completamente aleatorio: barajar y partir.

    No hay ninguna restricción sobre niveles; pueden formarse grupos
    desequilibrados (varios nivel 1 juntos, o ningún nivel 4, etc.).
    """
    if len(teams) != NUM_GROUPS * TEAMS_PER_GROUP:
        raise ValueError(
            f"Sorteo aleatorio requiere {NUM_GROUPS * TEAMS_PER_GROUP} equipos, "
            f"got {len(teams)}"
        )
    shuffled = teams.copy()
    rng.shuffle(shuffled)
    groups = [
        shuffled[i * TEAMS_PER_GROUP:(i + 1) * TEAMS_PER_GROUP]
        for i in range(NUM_GROUPS)
    ]
    return groups


def make_draw(draw_type: str, teams: List[Team],
              rng: np.random.Generator) -> List[List[Team]]:
    """Despachador. ``draw_type`` ∈ {``"balanced"``, ``"random"``}."""
    if draw_type == "balanced":
        return balanced_draw(teams, rng)
    if draw_type == "random":
        return random_draw(teams, rng)
    raise ValueError(f"draw_type desconocido: {draw_type!r} (use 'balanced' o 'random')")
