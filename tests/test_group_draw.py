"""Tests del sorteo de grupos."""
import numpy as np
from src.teams import create_teams
from src.group_draw import balanced_draw, random_draw


def test_balanced_draw_exactly_one_top_seed_per_group():
    """Sorteo balanceado: cada grupo lleva exactamente 1 equipo de nivel 1.
    Los demás niveles pueden repetirse o faltar dentro del grupo.
    """
    teams = create_teams()
    rng = np.random.default_rng(42)
    groups = balanced_draw(teams, rng)
    assert len(groups) == 8
    for g in groups:
        assert len(g) == 4
        n_level1 = sum(1 for t in g if t.level == 1)
        assert n_level1 == 1, f"Esperaba 1 nivel 1 por grupo, got {n_level1}"


def test_balanced_draw_distributes_other_levels_uniformly():
    """Los equipos no-nivel-1 deberían poder caer en cualquier grupo.
    Con muchos sorteos, cada equipo (id, group) debe tener algún caso.
    """
    teams = create_teams()
    counts = {(t.id, g_idx): 0 for t in teams for g_idx in range(8)}
    rng = np.random.default_rng(0)
    n = 4000
    for _ in range(n):
        groups = balanced_draw(teams, rng)
        for g_idx, g in enumerate(groups):
            for t in g:
                counts[(t.id, g_idx)] += 1
    for v in counts.values():
        assert v > 0
    # Equipos de nivel 1 deben aparecer ~n/8 veces en cada grupo (uniforme).
    teams_l1 = [t.id for t in teams if t.level == 1]
    for tid in teams_l1:
        for g_idx in range(8):
            # Holgura: 0.4 * n/8 < count < 1.6 * n/8 con seguridad
            assert 0.6 * (n / 8) < counts[(tid, g_idx)] < 1.4 * (n / 8)


def test_random_draw_size():
    teams = create_teams()
    rng = np.random.default_rng(42)
    groups = random_draw(teams, rng)
    assert len(groups) == 8
    assert all(len(g) == 4 for g in groups)


def test_no_duplicate_teams_balanced():
    teams = create_teams()
    rng = np.random.default_rng(42)
    groups = balanced_draw(teams, rng)
    all_ids = [t.id for g in groups for t in g]
    assert len(all_ids) == 32
    assert len(set(all_ids)) == 32


def test_no_duplicate_teams_random():
    teams = create_teams()
    rng = np.random.default_rng(42)
    groups = random_draw(teams, rng)
    all_ids = [t.id for g in groups for t in g]
    assert len(all_ids) == 32
    assert len(set(all_ids)) == 32


def test_balanced_reproducibility():
    teams = create_teams()
    g1 = balanced_draw(teams, np.random.default_rng(123))
    g2 = balanced_draw(teams, np.random.default_rng(123))
    for a, b in zip(g1, g2):
        assert [t.id for t in a] == [t.id for t in b]


def test_random_reproducibility():
    teams = create_teams()
    g1 = random_draw(teams, np.random.default_rng(123))
    g2 = random_draw(teams, np.random.default_rng(123))
    for a, b in zip(g1, g2):
        assert [t.id for t in a] == [t.id for t in b]


def test_balanced_distributes_each_level_team_uniformly():
    """Cada equipo debe poder caer en cualquier grupo (test estadístico ligero).
    Con 5_000 sorteos balanceados, cada equipo debería aparecer en cada grupo
    con frecuencia ~5000/8 = 625. Comprobamos que ningún grupo queda 'vacío'.
    """
    teams = create_teams()
    counts = {(t.id, g_idx): 0 for t in teams for g_idx in range(8)}
    rng = np.random.default_rng(0)
    n = 5000
    for _ in range(n):
        groups = balanced_draw(teams, rng)
        for g_idx, g in enumerate(groups):
            for t in g:
                counts[(t.id, g_idx)] += 1
    # Todos los pares (equipo, grupo) deberían tener algún caso
    for v in counts.values():
        assert v > 0
