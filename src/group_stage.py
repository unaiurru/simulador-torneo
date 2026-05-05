"""Simulación de la fase de grupos (round-robin de un solo turno).

Sistema de puntos estándar: V=3, E=1, D=0.

Desempate:
    1. Más puntos.
    2. Más victorias.
    3. Desempate aleatorio reproducible (gestionado por el ``Generator``).

No simulamos goles; el desempate (3) actúa como sustituto de la diferencia de
goles para mantener el modelo simple e interpretable. La aleatoriedad está
controlada por la semilla del Monte Carlo, así que sigue siendo reproducible.
"""
from itertools import combinations
from dataclasses import dataclass
from typing import List
import numpy as np

from .teams import Team
from .match_simulator import simulate_match, Outcome
from .probabilities import ProbabilityMatrix


POINTS_WIN = 3
POINTS_DRAW = 1
POINTS_LOSS = 0


@dataclass
class TeamStats:
    """Estadística acumulada de un equipo dentro de un grupo."""
    team: Team
    points: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    def add_outcome(self, outcome: Outcome, is_team_a: bool) -> None:
        """Actualiza V/E/D y puntos según el resultado del partido."""
        if outcome == Outcome.DRAW:
            self.draws += 1
            self.points += POINTS_DRAW
            return
        won = (outcome == Outcome.WIN_A and is_team_a) or \
              (outcome == Outcome.WIN_B and not is_team_a)
        if won:
            self.wins += 1
            self.points += POINTS_WIN
        else:
            self.losses += 1


def simulate_group(group: List[Team],
                   matrix: ProbabilityMatrix,
                   rng: np.random.Generator) -> List[TeamStats]:
    """Simula un grupo round-robin y devuelve la clasificación final.

    Devuelve una lista de ``TeamStats`` ordenada de 1º a 4º.
    """
    stats = {t.id: TeamStats(team=t) for t in group}

    for team_a, team_b in combinations(group, 2):
        outcome = simulate_match(team_a, team_b, matrix, rng)
        stats[team_a.id].add_outcome(outcome, is_team_a=True)
        stats[team_b.id].add_outcome(outcome, is_team_a=False)

    standings = list(stats.values())
    # Desempate aleatorio reproducible: barajamos antes de ordenar de forma
    # estable. Equipos con misma (puntos, victorias) quedan en orden aleatorio
    # pero determinista dado el rng.
    rng.shuffle(standings)
    standings.sort(key=lambda s: (s.points, s.wins), reverse=True)
    return standings


def simulate_all_groups(groups: List[List[Team]],
                        matrix: ProbabilityMatrix,
                        rng: np.random.Generator) -> List[List[TeamStats]]:
    """Simula todos los grupos. Devuelve lista de clasificaciones (una por grupo)."""
    return [simulate_group(g, matrix, rng) for g in groups]
