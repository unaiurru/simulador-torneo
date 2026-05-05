"""Definición y construcción de equipos.

Decisiones de diseño:
- Los equipos se identifican por (id, name, level). El nombre es puramente cosmético:
  toda la lógica de simulación usa exclusivamente `level`. Esto evita por construcción
  cualquier sesgo accidental por nombre o posición.
- Equipos del mismo nivel son intercambiables: las probabilidades dependen solo del
  nivel, por lo que la varianza entre equipos de un mismo nivel proviene únicamente
  del azar de la simulación.
"""
from dataclasses import dataclass
from typing import List


# Constantes de la competición
NUM_LEVELS = 4
TEAMS_PER_LEVEL = 8
NUM_GROUPS = 8
TEAMS_PER_GROUP = 4
TOTAL_TEAMS = NUM_LEVELS * TEAMS_PER_LEVEL  # 32

LEVEL_NAMES = {
    1: "Los mejores",
    2: "Los decentes",
    3: "Los no tan malos",
    4: "Los malos",
}


@dataclass(frozen=True)
class Team:
    """Equipo participante.

    Atributos:
        id: identificador entero único (0..31).
        name: nombre cosmético, no influye en la lógica.
        level: nivel del equipo (1 = mejor, 4 = peor).
    """
    id: int
    name: str
    level: int

    def __repr__(self) -> str:
        return f"{self.name}(N{self.level})"


def create_teams(teams_per_level: int = TEAMS_PER_LEVEL,
                 num_levels: int = NUM_LEVELS) -> List[Team]:
    """Crea el conjunto estándar de equipos: `teams_per_level` equipos por nivel.

    Por defecto devuelve 32 equipos (8 niveles × 4 niveles), nombrados como
    T<level><letra> (T1A, T1B, ..., T4H). El nombre es solo cosmético.
    """
    if teams_per_level < 1 or num_levels < 1:
        raise ValueError("teams_per_level y num_levels deben ser >= 1")

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if teams_per_level > len(letters):
        raise ValueError(
            f"teams_per_level={teams_per_level} excede letras disponibles ({len(letters)})"
        )

    teams: List[Team] = []
    team_id = 0
    for level in range(1, num_levels + 1):
        for i in range(teams_per_level):
            name = f"T{level}{letters[i]}"
            teams.append(Team(id=team_id, name=name, level=level))
            team_id += 1
    return teams
