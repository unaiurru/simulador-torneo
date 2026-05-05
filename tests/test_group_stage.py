"""Tests de la fase de grupos."""
import numpy as np
from src.teams import create_teams
from src.probabilities import ProbabilityMatrix
from src.group_draw import balanced_draw
from src.group_stage import simulate_group, simulate_all_groups, POINTS_WIN, POINTS_DRAW


def test_each_group_has_4_teams_in_standings():
    teams = create_teams()
    matrix = ProbabilityMatrix.from_default()
    rng = np.random.default_rng(42)
    groups = balanced_draw(teams, rng)
    standings = simulate_all_groups(groups, matrix, rng)
    assert len(standings) == 8
    for s in standings:
        assert len(s) == 4


def test_standings_sorted_by_points_then_wins():
    teams = create_teams()
    matrix = ProbabilityMatrix.from_default()
    rng = np.random.default_rng(42)
    groups = balanced_draw(teams, rng)
    standings = simulate_all_groups(groups, matrix, rng)
    for s in standings:
        for i in range(len(s) - 1):
            cur, nxt = s[i], s[i + 1]
            assert (cur.points, cur.wins) >= (nxt.points, nxt.wins)


def test_no_duplicate_teams_across_standings():
    teams = create_teams()
    matrix = ProbabilityMatrix.from_default()
    rng = np.random.default_rng(42)
    groups = balanced_draw(teams, rng)
    standings = simulate_all_groups(groups, matrix, rng)
    all_ids = [s.team.id for grp in standings for s in grp]
    assert len(all_ids) == 32
    assert len(set(all_ids)) == 32


def test_each_team_plays_three_matches():
    teams = create_teams()
    matrix = ProbabilityMatrix.from_default()
    rng = np.random.default_rng(42)
    groups = balanced_draw(teams, rng)
    standings = simulate_all_groups(groups, matrix, rng)
    for s in standings:
        for stat in s:
            assert stat.wins + stat.draws + stat.losses == 3


def test_points_consistent_with_outcomes():
    teams = create_teams()
    matrix = ProbabilityMatrix.from_default()
    rng = np.random.default_rng(42)
    groups = balanced_draw(teams, rng)
    standings = simulate_all_groups(groups, matrix, rng)
    for s in standings:
        for stat in s:
            assert stat.points == stat.wins * POINTS_WIN + stat.draws * POINTS_DRAW


def test_total_points_per_group_consistent():
    """En cada partido se reparten 2 (empate) o 3 (con ganador) puntos.
    Con 6 partidos por grupo, los puntos totales están en [12, 18]."""
    teams = create_teams()
    matrix = ProbabilityMatrix.from_default()
    rng = np.random.default_rng(42)
    groups = balanced_draw(teams, rng)
    standings = simulate_all_groups(groups, matrix, rng)
    for s in standings:
        total_pts = sum(stat.points for stat in s)
        assert 12 <= total_pts <= 18


def test_reproducibility():
    teams = create_teams()
    matrix = ProbabilityMatrix.from_default()

    def run():
        rng = np.random.default_rng(7)
        groups = balanced_draw(teams, rng)
        return simulate_all_groups(groups, matrix, rng)

    a = run()
    b = run()
    for ga, gb in zip(a, b):
        assert [s.team.id for s in ga] == [s.team.id for s in gb]
        assert [s.points for s in ga] == [s.points for s in gb]
