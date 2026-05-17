"""Modelo analítico cerrado del Teorema 2 (sección didáctica).

Este módulo NO usa Monte Carlo. Provee una aproximación analítica (sigmoide)
para ilustrar visualmente las tres propiedades del Teorema 2 del documento:

1.  La curva i ↦ P_i es una sigmoide cuyo punto de inflexión está en P = 1/2.
2.  P^ale_i = P^bal_i para todo i  ⟺  todas las fuerzas son iguales
    (o, equivalentemente en este modelo cerrado, la pendiente k = 0).
3.  (1/n) Σ P_i = 1/2  para fuerzas simétricas alrededor de 0.

Modelo
------
Cada equipo i tiene fuerza s_i ∈ R. La probabilidad de clasificar se modela como

        P_i = σ( k · (s_i − μ_rival_i) )

donde μ_rival_i es la fuerza media de un rival típico bajo el sorteo en cuestión:

  - Aleatorio puro:  μ_rival_i = (Σ s − s_i) / (n − 1).
  - Balanceado:      μ_rival_i = media de los (m − 1) bombos distintos al del equipo.

Limitaciones (intencionadas, declaradas)
----------------------------------------
- El modelo agrega la fuerza rival ANTES del sigmoide. La simulación real aplica
  la sigmoide partido a partido y luego suma, lo que activa Jensen al máximo.
- Por tanto, este módulo subestima la magnitud (no el signo) de la separación
  entre curvas. Sigue siendo correcto como demostración cualitativa del Teorema.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np


Mode = Literal["random", "balanced"]


# --------------------------------------------------------------------------- #
# 1. Sigmoide y derivadas
# --------------------------------------------------------------------------- #
def sigmoid(x: np.ndarray | float, k: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """Sigmoide logística σ(x) = 1 / (1 + e^{-k(x-x0)}).

    Parámetros
    ----------
    x  : escalar o array. Variable independiente (ventaja de fuerza).
    k  : pendiente. k → 0 aplana la curva; k = 0 produce la constante 1/2.
    x0 : punto de inflexión (P = 1/2). Por defecto x0 = 0.
    """
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(x) - x0)))


def sigmoid_derivative(x, k: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """σ'(x) = k · σ(x) · (1 − σ(x))."""
    s = sigmoid(x, k=k, x0=x0)
    return k * s * (1.0 - s)


def sigmoid_second_derivative(x, k: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """σ''(x) = k² · σ(x) · (1 − σ(x)) · (1 − 2σ(x)).

    Cambia de signo exactamente en σ = 1/2 (punto de inflexión).
    """
    s = sigmoid(x, k=k, x0=x0)
    return (k ** 2) * s * (1.0 - s) * (1.0 - 2.0 * s)


# --------------------------------------------------------------------------- #
# 2. Modelo de fuerzas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TheoryConfig:
    """Parámetros del modelo analítico cerrado."""
    n: int = 32
    m: int = 4              # tamaño de grupo
    spread: float = 1.0     # heterogeneidad de fuerzas (0 = todas iguales)
    steepness: float = 2.5  # k en la sigmoide


def generate_strengths(n: int, spread: float) -> np.ndarray:
    """Devuelve un array de n fuerzas ordenadas DECRECIENTES.

    - El equipo 0 (rango 1) es el más fuerte; el equipo n-1 el más débil.
    - Las fuerzas son simétricas alrededor de 0 → promedio teórico exacto 1/2.
    - spread = 0 ⇒ todas iguales (caso ⇐ del Teorema 2).
    """
    if spread <= 0:
        return np.zeros(n)
    return np.linspace(spread, -spread, n)


def _bombo_means(strengths_sorted: np.ndarray, m: int) -> np.ndarray:
    """Promedio de fuerza dentro de cada uno de los m bombos.

    `strengths_sorted` debe estar ordenado de mayor a menor: los primeros n/m
    forman el bombo 1, los siguientes el bombo 2, etc.
    """
    n = len(strengths_sorted)
    if n % m != 0:
        raise ValueError(f"n={n} no es divisible por m={m}")
    bombo_size = n // m
    return np.array([
        strengths_sorted[b * bombo_size:(b + 1) * bombo_size].mean()
        for b in range(m)
    ])


# --------------------------------------------------------------------------- #
# 3. Curvas teóricas P_ale y P_bal
# --------------------------------------------------------------------------- #
def expected_rival_strength_random(strengths: np.ndarray) -> np.ndarray:
    """μ_rival aleatorio puro: media de fuerza de cualquier otro equipo."""
    n = len(strengths)
    if n < 2:
        return np.zeros_like(strengths)
    total = strengths.sum()
    return (total - strengths) / (n - 1)


def expected_rival_strength_balanced(strengths: np.ndarray, m: int = 4) -> np.ndarray:
    """μ_rival balanceado: media de los m-1 bombos distintos al del equipo.

    Asume `strengths` ordenado decreciente (los primeros n/m → bombo 1, etc.).
    """
    n = len(strengths)
    bombo_size = n // m
    bombo_mu = _bombo_means(strengths, m)
    out = np.zeros(n)
    for i in range(n):
        my_bombo = i // bombo_size
        others = [b for b in range(m) if b != my_bombo]
        out[i] = bombo_mu[others].mean()
    return out


def qualification_curve(
    strengths: np.ndarray,
    steepness: float,
    mode: Mode,
    m: int = 4,
) -> np.ndarray:
    """Curva analítica P_i = σ(k · (s_i − μ_rival_i)).

    `mode` ∈ {"random", "balanced"}. La devolución tiene shape (n,).
    """
    if mode == "random":
        mu = expected_rival_strength_random(strengths)
    elif mode == "balanced":
        mu = expected_rival_strength_balanced(strengths, m=m)
    else:
        raise ValueError(f"Modo desconocido: {mode!r}. Usa 'random' o 'balanced'.")
    return sigmoid(strengths - mu, k=steepness, x0=0.0)


# --------------------------------------------------------------------------- #
# 4. Utilidades para visualización
# --------------------------------------------------------------------------- #
def find_crossings(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Índices i tales que el signo de (p_a − p_b) cambia entre i e i+1."""
    diff = np.asarray(p_a) - np.asarray(p_b)
    signs = np.sign(diff)
    # Ignorar ceros exactos: tratarlos como continuación del signo anterior
    return np.where(np.diff(signs) != 0)[0]


def average_probability(probs: np.ndarray) -> float:
    """Promedio empírico de las probabilidades (debería ser ≈ 1/2)."""
    return float(np.mean(probs))


def curves_are_equal(
    strengths: np.ndarray,
    steepness: float,
    m: int = 4,
    atol: float = 1e-9,
) -> bool:
    """True si P^ale y P^bal coinciden punto a punto (caso del Teorema 2)."""
    p_ale = qualification_curve(strengths, steepness, "random", m=m)
    p_bal = qualification_curve(strengths, steepness, "balanced", m=m)
    return bool(np.allclose(p_ale, p_bal, atol=atol))
