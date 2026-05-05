"""Matriz de probabilidades entre niveles.

La matriz almacena, para cada par de niveles (i, j), la probabilidad de que un
equipo de nivel i obtenga victoria, empate o derrota al enfrentarse a uno de
nivel j.

Coherencia obligatoria (validada en runtime):
    - P(victoria_i_vs_j) + P(empate_i,j) + P(derrota_i_vs_j) = 1
    - P(victoria_i_vs_j) = P(derrota_j_vs_i)
    - P(empate_i,j)    = P(empate_j,i)
    - Todas las probabilidades deben estar en [0, 1].

Toda la lógica del proyecto consume probabilidades a través de esta clase, lo
que centraliza la fuente de verdad y permite editarlas desde el dashboard sin
tocar el resto del código.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Valores por defecto (los proporcionados por el usuario)
# ---------------------------------------------------------------------------
# Formato: (p_victoria_i_vs_j, p_empate, p_derrota_i_vs_j)
DEFAULT_PROBABILITIES: Dict[Tuple[int, int], Tuple[float, float, float]] = {
    (1, 1): (0.35, 0.30, 0.35),
    (1, 2): (0.55, 0.25, 0.20),
    (1, 3): (0.70, 0.20, 0.10),
    (1, 4): (0.82, 0.13, 0.05),
    (2, 2): (0.35, 0.30, 0.35),
    (2, 3): (0.55, 0.25, 0.20),
    (2, 4): (0.70, 0.20, 0.10),
    (3, 3): (0.35, 0.30, 0.35),
    (3, 4): (0.58, 0.24, 0.18),
    (4, 4): (0.35, 0.30, 0.35),
}


@dataclass
class ProbabilityMatrix:
    """Matriz simétrica de probabilidades por pares de niveles.

    Internamente guarda tres matrices NxN (`win`, `draw`, `loss`) donde N es
    el número de niveles. `set_probs(i, j, w, d, l)` actualiza también las
    entradas (j, i) para mantener simetría automática.
    """
    num_levels: int = 4
    win: np.ndarray = field(init=False)
    draw: np.ndarray = field(init=False)
    loss: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = self.num_levels
        self.win = np.zeros((n, n), dtype=float)
        self.draw = np.zeros((n, n), dtype=float)
        self.loss = np.zeros((n, n), dtype=float)

    # ------------------------------------------------------------------
    # Constructores
    # ------------------------------------------------------------------
    @classmethod
    def from_default(cls) -> "ProbabilityMatrix":
        """Devuelve la matriz con los valores por defecto del proyecto."""
        return cls.from_dict(DEFAULT_PROBABILITIES, num_levels=4)

    @classmethod
    def from_dict(cls,
                  probs: Dict[Tuple[int, int], Tuple[float, float, float]],
                  num_levels: int = 4) -> "ProbabilityMatrix":
        """Construye la matriz a partir de un diccionario y la valida.

        El diccionario solo necesita contener los pares (i, j) con i <= j.
        Las entradas espejo se rellenan automáticamente.
        """
        m = cls(num_levels=num_levels)
        seen = set()
        for (i, j), (w, d, l) in probs.items():
            key = tuple(sorted((i, j)))
            if key in seen:
                # ya cubierto por simetría
                continue
            seen.add(key)
            m.set_probs(i, j, w, d, l)
        m.validate()
        return m

    # ------------------------------------------------------------------
    # Acceso y mutación
    # ------------------------------------------------------------------
    def set_probs(self, level_i: int, level_j: int,
                  p_win: float, p_draw: float, p_loss: float) -> None:
        """Actualiza la entrada (i, j) y la espejo (j, i) automáticamente."""
        i, j = level_i - 1, level_j - 1
        self.win[i, j] = p_win
        self.draw[i, j] = p_draw
        self.loss[i, j] = p_loss
        # Espejo: la victoria de i contra j es la derrota de j contra i.
        self.win[j, i] = p_loss
        self.draw[j, i] = p_draw
        self.loss[j, i] = p_win

    def get_probs(self, level_i: int, level_j: int) -> Tuple[float, float, float]:
        """Devuelve (p_victoria, p_empate, p_derrota) para nivel i contra nivel j."""
        i, j = level_i - 1, level_j - 1
        return (float(self.win[i, j]), float(self.draw[i, j]), float(self.loss[i, j]))

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------
    def validate(self, tol: float = 1e-9) -> None:
        """Valida coherencia. Lanza ValueError con mensaje claro si falla."""
        n = self.num_levels
        for i in range(n):
            for j in range(n):
                w, d, l = self.win[i, j], self.draw[i, j], self.loss[i, j]
                if any(p < -tol or p > 1 + tol for p in (w, d, l)):
                    raise ValueError(
                        f"Probabilidades fuera de [0, 1] en niveles {i+1} vs {j+1}: "
                        f"({w:.3f}, {d:.3f}, {l:.3f})"
                    )
                s = w + d + l
                if not np.isclose(s, 1.0, atol=1e-6):
                    raise ValueError(
                        f"Las probabilidades en niveles {i+1} vs {j+1} suman {s:.4f} "
                        f"(deben sumar 1)"
                    )
                if not np.isclose(self.draw[i, j], self.draw[j, i], atol=tol):
                    raise ValueError(
                        f"Empate no simétrico entre niveles {i+1} y {j+1}: "
                        f"{self.draw[i, j]:.4f} vs {self.draw[j, i]:.4f}"
                    )
                if not np.isclose(self.win[i, j], self.loss[j, i], atol=tol):
                    raise ValueError(
                        f"P(victoria {i+1}vs{j+1}) debería igualar P(derrota {j+1}vs{i+1}): "
                        f"{self.win[i, j]:.4f} vs {self.loss[j, i]:.4f}"
                    )

    # ------------------------------------------------------------------
    # Presentación
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve la matriz como DataFrame legible (todos los pares ordenados)."""
        rows = []
        for i in range(1, self.num_levels + 1):
            for j in range(i, self.num_levels + 1):
                w, d, l = self.get_probs(i, j)
                rows.append({
                    "Nivel A": i,
                    "Nivel B": j,
                    "P(victoria A)": round(w, 4),
                    "P(empate)": round(d, 4),
                    "P(victoria B)": round(l, 4),
                })
        return pd.DataFrame(rows)
