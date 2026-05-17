"""Tests del módulo teórico (Teorema 2 y propiedades de la sigmoide)."""
import numpy as np
import pytest

from src import theory


# --------------------------------------------------------------------------- #
# Sigmoide y derivadas
# --------------------------------------------------------------------------- #
class TestSigmoid:
    def test_sigmoid_at_zero_is_half(self):
        assert np.isclose(theory.sigmoid(0.0), 0.5)

    def test_sigmoid_zero_steepness_is_flat_one_half(self):
        x = np.linspace(-5, 5, 50)
        y = theory.sigmoid(x, k=0.0)
        assert np.allclose(y, 0.5)

    def test_sigmoid_monotone_increasing(self):
        x = np.linspace(-5, 5, 100)
        y = theory.sigmoid(x, k=2.0)
        assert np.all(np.diff(y) >= 0)

    def test_inflection_point_at_half(self):
        # σ'' = 0 exactamente cuando σ = 1/2 (en x = x0 = 0)
        assert np.isclose(theory.sigmoid_second_derivative(0.0, k=1.0), 0.0)

    def test_curvature_sign_changes_at_half(self):
        # σ'' > 0 si σ < 1/2 (convexa); σ'' < 0 si σ > 1/2 (cóncava)
        assert theory.sigmoid_second_derivative(-1.0, k=1.0) > 0
        assert theory.sigmoid_second_derivative(+1.0, k=1.0) < 0

    def test_derivative_positive(self):
        # σ'(x) > 0 para k > 0 (sigmoide creciente)
        for xi in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            assert theory.sigmoid_derivative(xi, k=1.5) > 0


# --------------------------------------------------------------------------- #
# Modelo de fuerzas
# --------------------------------------------------------------------------- #
class TestStrengths:
    def test_generate_strengths_descending(self):
        s = theory.generate_strengths(32, spread=1.0)
        assert s[0] > s[-1]
        assert np.all(np.diff(s) <= 0)

    def test_generate_strengths_symmetric_around_zero(self):
        s = theory.generate_strengths(32, spread=1.0)
        assert np.isclose(s.sum(), 0.0)

    def test_generate_strengths_zero_spread_is_constant(self):
        s = theory.generate_strengths(32, spread=0.0)
        assert np.allclose(s, 0.0)


# --------------------------------------------------------------------------- #
# Teorema 2: equivalencia de sorteos
# --------------------------------------------------------------------------- #
class TestTheorem:
    def test_equal_strengths_implies_equal_curves(self):
        """⇐: spread = 0 ⇒ P_ale = P_bal = 1/2 para todos."""
        s = theory.generate_strengths(32, spread=0.0)
        p_ale = theory.qualification_curve(s, steepness=2.0, mode="random")
        p_bal = theory.qualification_curve(s, steepness=2.0, mode="balanced")
        assert np.allclose(p_ale, p_bal)
        assert np.allclose(p_ale, 0.5)

    def test_zero_steepness_collapses_to_one_half(self):
        """k = 0 produce ambas curvas constantes en 1/2 (modelo trivial)."""
        s = theory.generate_strengths(32, spread=1.0)
        p_ale = theory.qualification_curve(s, steepness=0.0, mode="random")
        p_bal = theory.qualification_curve(s, steepness=0.0, mode="balanced")
        assert np.allclose(p_ale, 0.5)
        assert np.allclose(p_bal, 0.5)

    def test_average_is_one_half_for_symmetric_strengths(self):
        """Proposición 1: promedio teórico = r/m = 1/2."""
        s = theory.generate_strengths(32, spread=1.0)
        for mode in ("random", "balanced"):
            p = theory.qualification_curve(s, steepness=2.0, mode=mode)
            assert np.isclose(np.mean(p), 0.5, atol=1e-9), (
                f"Promedio incorrecto en modo {mode}: {np.mean(p)}"
            )

    def test_heterogeneous_curves_differ(self):
        """⇒: con heterogeneidad y k > 0, las curvas NO coinciden."""
        s = theory.generate_strengths(32, spread=1.0)
        p_ale = theory.qualification_curve(s, steepness=2.0, mode="random")
        p_bal = theory.qualification_curve(s, steepness=2.0, mode="balanced")
        assert not np.allclose(p_ale, p_bal)

    def test_balanced_helps_strong_and_hurts_weak(self):
        """Patrón cualitativo (PDF §4): top P_bal > P_ale; bottom P_bal < P_ale."""
        s = theory.generate_strengths(32, spread=1.0)
        p_ale = theory.qualification_curve(s, steepness=2.5, mode="random")
        p_bal = theory.qualification_curve(s, steepness=2.5, mode="balanced")
        # Equipo más fuerte (rango 1, índice 0)
        assert p_bal[0] > p_ale[0]
        # Equipo más débil (rango n, índice -1)
        assert p_bal[-1] < p_ale[-1]

    def test_at_least_one_crossing_when_heterogeneous(self):
        """Las dos curvas se cruzan al menos una vez (consecuencia operativa §5)."""
        s = theory.generate_strengths(32, spread=1.0)
        p_ale = theory.qualification_curve(s, steepness=2.0, mode="random")
        p_bal = theory.qualification_curve(s, steepness=2.0, mode="balanced")
        crossings = theory.find_crossings(p_bal, p_ale)
        assert len(crossings) >= 1


# --------------------------------------------------------------------------- #
# Reproducibilidad y robustez
# --------------------------------------------------------------------------- #
class TestRobustness:
    def test_qualification_curve_is_deterministic(self):
        s = theory.generate_strengths(32, spread=1.0)
        p1 = theory.qualification_curve(s, steepness=2.0, mode="random")
        p2 = theory.qualification_curve(s, steepness=2.0, mode="random")
        assert np.array_equal(p1, p2)

    def test_unknown_mode_raises(self):
        s = theory.generate_strengths(32, spread=1.0)
        with pytest.raises(ValueError):
            theory.qualification_curve(s, steepness=2.0, mode="other")  # type: ignore[arg-type]

    def test_curves_are_equal_helper(self):
        s_eq = theory.generate_strengths(32, spread=0.0)
        s_neq = theory.generate_strengths(32, spread=1.0)
        assert theory.curves_are_equal(s_eq, steepness=2.0)
        assert not theory.curves_are_equal(s_neq, steepness=2.0)
