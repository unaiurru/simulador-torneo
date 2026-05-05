"""Fase final eliminatoria (octavos → cuartos → semis → final).

Cuadro de octavos:
    - Los 8 primeros de grupo se enfrentan a los 8 segundos.
    - Se intenta evitar que dos equipos del mismo grupo se crucen en R16
      reordenando aleatoriamente la lista de segundos hasta encontrar una
      permutación válida (con un máximo de intentos para evitar loops).

Bracket posterior fijo: los emparejamientos de cuartos, semis y final se
forman por orden (R16[0]-R16[1] → cuartos[0], etc.). Esto mantiene el modelo
simple. Al ser los grupos equivalentes, el resultado agregado no se ve afectado.
"""
from typing import Dict, List, Tuple
import numpy as np

from .teams import Team
from .group_stage import TeamStats
from .match_simulator import simulate_knockout_match
from .probabilities import ProbabilityMatrix


# Orden canónico de las rondas. Mayor índice = más lejos en el torneo.
ROUND_ORDER: List[str] = [
    "did_not_qualify",  # 0  - eliminado en fase de grupos
    "round_of_16",      # 1
    "quarterfinals",    # 2
    "semifinals",       # 3
    "final",            # 4
    "champion",         # 5
]
ROUND_RANK: Dict[str, int] = {r: i for i, r in enumerate(ROUND_ORDER)}


def build_bracket(group_standings: List[List[TeamStats]],
                  rng: np.random.Generator,
                  max_attempts: int = 200) -> List[Tuple[Team, Team]]:
    """Construye los emparejamientos de octavos.

    Empareja primeros vs segundos intentando evitar enfrentamientos del mismo
    grupo. Si tras ``max_attempts`` no se logra una permutación válida (caso
    extremadamente raro), se acepta la última.

    Devuelve una lista de 8 tuplas (1º, 2º).
    """
    if len(group_standings) != 8:
        raise ValueError(f"Se esperaban 8 grupos, got {len(group_standings)}")

    firsts: List[Team] = [s[0].team for s in group_standings]
    seconds: List[Team] = [s[1].team for s in group_standings]

    # Mapa equipo_id -> índice de grupo
    team_group: Dict[int, int] = {}
    for g_idx, standings in enumerate(group_standings):
        for s in standings:
            team_group[s.team.id] = g_idx

    seconds_perm = seconds.copy()
    for _ in range(max_attempts):
        rng.shuffle(seconds_perm)
        if all(team_group[firsts[i].id] != team_group[seconds_perm[i].id]
               for i in range(8)):
            break

    return list(zip(firsts, seconds_perm))


def simulate_knockout(bracket: List[Tuple[Team, Team]],
                      matrix: ProbabilityMatrix,
                      rng: np.random.Generator) -> Dict[int, str]:
    """Simula octavos → final. Devuelve dict {team_id -> ronda_máxima_alcanzada}.

    Solo aparecen los 16 equipos del cuadro; los no clasificados quedan fuera
    del diccionario (el orquestador les asignará 'did_not_qualify').
    """
    rounds_reached: Dict[int, str] = {}

    # ----- Octavos -----
    quarter_finalists: List[Team] = []
    for team_a, team_b in bracket:
        rounds_reached[team_a.id] = "round_of_16"
        rounds_reached[team_b.id] = "round_of_16"
        winner = simulate_knockout_match(team_a, team_b, matrix, rng)
        quarter_finalists.append(winner)

    # ----- Cuartos -----
    semi_finalists: List[Team] = []
    for i in range(0, 8, 2):
        a, b = quarter_finalists[i], quarter_finalists[i + 1]
        rounds_reached[a.id] = "quarterfinals"
        rounds_reached[b.id] = "quarterfinals"
        winner = simulate_knockout_match(a, b, matrix, rng)
        semi_finalists.append(winner)

    # ----- Semifinales -----
    finalists: List[Team] = []
    for i in range(0, 4, 2):
        a, b = semi_finalists[i], semi_finalists[i + 1]
        rounds_reached[a.id] = "semifinals"
        rounds_reached[b.id] = "semifinals"
        winner = simulate_knockout_match(a, b, matrix, rng)
        finalists.append(winner)

    # ----- Final -----
    a, b = finalists[0], finalists[1]
    rounds_reached[a.id] = "final"
    rounds_reached[b.id] = "final"
    champion = simulate_knockout_match(a, b, matrix, rng)
    rounds_reached[champion.id] = "champion"

    return rounds_reached
