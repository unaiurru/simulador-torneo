"""Simulación de un partido individual.

Hay dos primitivas:
- ``simulate_match``: partido de fase de grupos. Resultado ∈ {VICTORIA_A,
  EMPATE, VICTORIA_B} muestreado según la matriz de probabilidades.
- ``simulate_knockout_match``: partido eliminatorio. No admite empate; se
  reescala la probabilidad de victoria para excluir el caso de empate.

Ambos consumen un ``numpy.random.Generator`` para reproducibilidad.
"""
from enum import Enum
from typing import Tuple
import numpy as np

from .teams import Team
from .probabilities import ProbabilityMatrix


class Outcome(Enum):
    """Resultado posible de un partido de fase de grupos."""
    WIN_A = "win_a"
    DRAW = "draw"
    WIN_B = "win_b"


def simulate_match(team_a: Team, team_b: Team,
                   matrix: ProbabilityMatrix,
                   rng: np.random.Generator) -> Outcome:
    """Simula un partido y devuelve el resultado desde la perspectiva de A."""
    p_win, p_draw, _p_loss = matrix.get_probs(team_a.level, team_b.level)
    r = rng.random()
    if r < p_win:
        return Outcome.WIN_A
    if r < p_win + p_draw:
        return Outcome.DRAW
    return Outcome.WIN_B


def simulate_knockout_match(team_a: Team, team_b: Team,
                            matrix: ProbabilityMatrix,
                            rng: np.random.Generator) -> Team:
    """Simula un partido eliminatorio (sin empate) y devuelve el ganador.

    Estrategia: condicionamos a que no haya empate. Si las probabilidades
    base son (p_w, p_d, p_l), entonces P(A gana | no empate) = p_w / (p_w + p_l).

    Esto preserva la fuerza relativa entre equipos del modelo base sin
    introducir parámetros adicionales.
    """
    p_win, _p_draw, p_loss = matrix.get_probs(team_a.level, team_b.level)
    total = p_win + p_loss
    if total <= 0:
        # Caso patológico: si ambos solo pueden empatar, decidimos al 50/50
        return team_a if rng.random() < 0.5 else team_b
    p_a = p_win / total
    return team_a if rng.random() < p_a else team_b
