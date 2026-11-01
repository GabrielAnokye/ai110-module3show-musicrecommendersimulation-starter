"""Tests for the optional ranking modes (Strategy pattern bonus challenge)."""

import random
import re

import pytest

from experiments.weight_experiment import EXPERIMENT, weights
from src import recommender
from src.main import DATA_PATH, PROFILES, main
from src.recommender import (
    DEFAULT_MODE,
    STRATEGIES,
    Recommender,
    ScoringWeights,
    Song,
    UserProfile,
    get_strategy,
    load_songs,
    recommend_songs,
    score_song,
)

ALL_MODES = ["baseline", "mood-first", "energy-focused"]
POP = PROFILES["High-Energy Pop"]
ROCK = PROFILES["Intense Rock"]
ADVERSARIAL = PROFILES["Adversarial Listener"]


@pytest.fixture(scope="module")
def catalog():
    return load_songs(str(DATA_PATH))


def titles(results):
    return [song["title"] for song, _, _ in results]


def by_title(catalog, title):
    return next(song for song in catalog if song["title"] == title)


# ---------------------------------------------------------------- registry and weights

def test_three_modes_are_available_and_baseline_is_default():
    assert list(STRATEGIES) == ALL_MODES
    assert DEFAULT_MODE == "baseline"


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("baseline", ScoringWeights(genre=4.0, mood=3.0, energy=3.0, acoustic=1.0)),
        ("mood-first", ScoringWeights(genre=2.0, mood=4.0, energy=3.0, acoustic=1.0)),
        ("energy-focused", ScoringWeights(genre=2.0, mood=3.0, energy=6.0, acoustic=1.0)),
    ],
)
def test_each_mode_uses_its_weights(mode, expected):
    assert get_strategy(mode).weights() == expected


def test_max_scores():
    assert get_strategy("baseline").weights().max_score == recommender.MAX_SCORE == 11.0
    assert get_strategy("mood-first").weights().max_score == 10.0
    assert get_strategy("energy-focused").weights().max_score == 12.0


def test_weights_are_immutable():
    with pytest.raises(AttributeError):
        get_strategy("mood-first").weights().genre = 9.0


def test_unknown_or_non_string_mode_is_rejected(catalog):
    with pytest.raises(ValueError, match="unknown mode 'genre-first'"):
        recommend_songs(POP, catalog, mode="genre-first")
    with pytest.raises(ValueError, match="unknown mode"):
        score_song(POP, catalog[0], mode="Mood-First")
    with pytest.raises(TypeError):
        get_strategy(None)
    with pytest.raises(ValueError):
        Recommender([], mode="loudest")


# ---------------------------------------------------------------- baseline unchanged

@pytest.mark.parametrize("profile", list(PROFILES.values()), ids=list(PROFILES))
def test_baseline_mode_matches_default_calls(catalog, profile):
    default = recommend_songs(profile, catalog, k=18)
    explicit = recommend_songs(profile, catalog, k=18, mode="baseline")
    assert [(s["id"], score, why) for s, score, why in default] == [
        (s["id"], score, why) for s, score, why in explicit
    ]


def test_baseline_still_reads_patchable_module_constants(catalog):
    sunrise = by_title(catalog, "Sunrise City")
    with weights(EXPERIMENT):
        # The Phase 4 experiment patches the module constants: baseline follows them...
        assert score_song(POP, sunrise)[0] == pytest.approx(11.70)
        # ...while the fixed modes are unaffected.
        assert score_song(POP, sunrise, mode="mood-first")[0] == pytest.approx(9.76)
    assert score_song(POP, sunrise)[0] == pytest.approx(10.76)


def test_energy_focused_matches_phase4_experiment(catalog):
    for profile in PROFILES.values():
        with weights(EXPERIMENT):
            patched = recommend_songs(profile, catalog, k=18)
        mode = recommend_songs(profile, catalog, k=18, mode="energy-focused")
        assert [(s["id"], round(score, 9)) for s, score, _ in patched] == [
            (s["id"], round(score, 9)) for s, score, _ in mode
        ]


def test_weight_experiment_script_still_runs(capsys):
    from experiments import weight_experiment

    weight_experiment.main()
    out = capsys.readouterr().out
    assert "Weights after experiment: {'GENRE_WEIGHT': 4.0, 'MOOD_WEIGHT': 3.0" in out


# ---------------------------------------------------------------- modes change priorities

def test_mood_first_increases_mood_influence(catalog):
    golden_hour = by_title(catalog, "Golden Hour Lagos")  # afrobeats, happy
    base, _ = score_song(POP, golden_hour)
    mood_first, reasons = score_song(POP, golden_hour, mode="mood-first")
    assert mood_first - base == pytest.approx(1.0)
    assert "Mood match: happy (+4.00)" in reasons
    # Happy non-pop songs now outrank Gym Hero (pop, intense).
    assert titles(recommend_songs(POP, catalog))[:3] == ["Sunrise City", "Gym Hero", "Golden Hour Lagos"]
    assert titles(recommend_songs(POP, catalog, mode="mood-first"))[:3] == [
        "Sunrise City", "Golden Hour Lagos", "Rooftop Lights",
    ]
    # The baseline Gym Hero / Static Postcards tie is now decided by score, not by id.
    rock = recommend_songs(ROCK, catalog, mode="mood-first")
    assert [(s["title"], round(score, 2)) for s, score, _ in rock[2:4]] == [
        ("Gym Hero", 7.86), ("Static Postcards", 4.86),
    ]


def test_energy_focused_increases_energy_influence(catalog):
    storm_runner = by_title(catalog, "Storm Runner")  # energy 0.91 vs target 0.90
    _, reasons = score_song(ADVERSARIAL, storm_runner, mode="energy-focused")
    assert "Energy 0.91 vs target 0.90 (+5.94)" in reasons

    def mean_energy_gap(mode):
        top = recommend_songs(ADVERSARIAL, catalog, mode=mode)
        return sum(abs(s["energy"] - ADVERSARIAL["target_energy"]) for s, _, _ in top) / len(top)

    assert mean_energy_gap("energy-focused") < mean_energy_gap("baseline")
    assert titles(recommend_songs(ADVERSARIAL, catalog, mode="energy-focused"))[0] == "Storm Runner"


def test_each_mode_picks_a_different_top_song_for_the_adversarial_listener(catalog):
    tops = {mode: titles(recommend_songs(ADVERSARIAL, catalog, k=1, mode=mode))[0] for mode in ALL_MODES}
    assert tops == {
        "baseline": "Midnight Coding",      # genre wins
        "mood-first": "Paper Boats",        # mood wins
        "energy-focused": "Storm Runner",   # energy wins
    }


# ---------------------------------------------------------------- determinism and explanations

@pytest.mark.parametrize("mode", ALL_MODES)
def test_rankings_are_deterministic_in_every_mode(catalog, mode):
    for profile in PROFILES.values():
        expected = [s["id"] for s, _, _ in recommend_songs(profile, catalog, k=18, mode=mode)]
        shuffled = list(catalog)
        random.Random(7).shuffle(shuffled)
        assert [s["id"] for s, _, _ in recommend_songs(profile, shuffled, k=18, mode=mode)] == expected
        assert [s["id"] for s, _, _ in recommend_songs(profile, catalog, k=18, mode=mode)] == expected


@pytest.mark.parametrize("mode", ALL_MODES)
def test_explanations_add_up_to_the_score(catalog, mode):
    w = get_strategy(mode).weights()
    for profile in PROFILES.values():
        for song in catalog:
            score, reasons = score_song(profile, song, mode=mode)
            points = [float(p) for p in re.findall(r"\(\+(\d+\.\d\d)\)", " ".join(reasons))]
            assert len(points) == 4
            assert sum(points) == pytest.approx(score, abs=0.02)  # each part is shown to 2 decimals
            assert points[0] in (0.0, w.genre)
            assert points[1] in (0.0, w.mood)
            assert 0.0 <= points[2] <= w.energy and 0.0 <= points[3] <= w.acoustic


# ---------------------------------------------------------------- OOP interface and CLI

def test_recommender_class_supports_modes(catalog):
    songs = [Song(**song) for song in catalog]
    user = UserProfile(**ADVERSARIAL)
    assert Recommender(songs).mode == "baseline"
    for mode in ALL_MODES:
        rec = Recommender(songs, mode=mode)
        expected = [s["id"] for s, _, _ in recommend_songs(ADVERSARIAL, catalog, mode=mode)]
        assert [song.id for song in rec.recommend(user)] == expected
        _, reasons = score_song(ADVERSARIAL, catalog[0], mode=mode)
        assert rec.explain_recommendation(user, songs[0]) == "; ".join(reasons)


def test_cli_mode_baseline_output_matches_default(capsys):
    main([])
    default = capsys.readouterr().out
    main(["--mode", "baseline"])
    assert capsys.readouterr().out == default
    assert "Ranking mode" not in default


@pytest.mark.parametrize("mode", ["mood-first", "energy-focused"])
def test_cli_shows_non_default_mode(capsys, mode):
    main(["--all-profiles", "--mode", mode])
    out = capsys.readouterr().out
    assert out.count(f"Ranking mode: {mode}") == 1
    assert out.count("Because:") == 5 * len(PROFILES)


def test_cli_rejects_invalid_mode(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--mode", "loudest"])
    assert exit_info.value.code == 2
    assert "invalid choice: 'loudest'" in capsys.readouterr().err
