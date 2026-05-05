"""Tests de coherencia y validación de la matriz de probabilidades."""
import pytest
from src.probabilities import ProbabilityMatrix, DEFAULT_PROBABILITIES


def test_default_matrix_validates():
    m = ProbabilityMatrix.from_default()
    m.validate()


def test_probabilities_sum_to_one():
    m = ProbabilityMatrix.from_default()
    for i in range(1, 5):
        for j in range(1, 5):
            w, d, l = m.get_probs(i, j)
            assert abs((w + d + l) - 1.0) < 1e-9, (i, j, w, d, l)


def test_symmetry_win_loss():
    m = ProbabilityMatrix.from_default()
    for i in range(1, 5):
        for j in range(1, 5):
            w_ij, d_ij, l_ij = m.get_probs(i, j)
            w_ji, d_ji, l_ji = m.get_probs(j, i)
            assert abs(w_ij - l_ji) < 1e-9
            assert abs(d_ij - d_ji) < 1e-9
            assert abs(l_ij - w_ji) < 1e-9


def test_same_level_balanced():
    """Mismo nivel contra mismo nivel: P(victoria) == P(derrota)."""
    m = ProbabilityMatrix.from_default()
    for level in range(1, 5):
        w, d, l = m.get_probs(level, level)
        assert abs(w - l) < 1e-9, level


def test_invalid_sum_raises():
    bad = {(1, 1): (0.5, 0.5, 0.5)}  # suma 1.5
    with pytest.raises(ValueError, match="suman"):
        ProbabilityMatrix.from_dict(bad, num_levels=1)


def test_negative_probability_raises():
    bad = {(1, 1): (-0.1, 0.5, 0.6)}
    with pytest.raises(ValueError):
        ProbabilityMatrix.from_dict(bad, num_levels=1)


def test_set_probs_keeps_symmetry():
    m = ProbabilityMatrix.from_default()
    m.set_probs(1, 4, 0.9, 0.05, 0.05)
    w14, d14, l14 = m.get_probs(1, 4)
    w41, d41, l41 = m.get_probs(4, 1)
    assert (w14, d14, l14) == (0.9, 0.05, 0.05)
    assert (w41, d41, l41) == (0.05, 0.05, 0.9)
    m.validate()


def test_default_values_match_spec():
    """Los valores por defecto deben coincidir con los de la especificación."""
    m = ProbabilityMatrix.from_default()
    expected = DEFAULT_PROBABILITIES
    for (i, j), (w, d, l) in expected.items():
        got_w, got_d, got_l = m.get_probs(i, j)
        assert abs(got_w - w) < 1e-9
        assert abs(got_d - d) < 1e-9
        assert abs(got_l - l) < 1e-9
