"""Tests for the Phase 4 CLI profiles and the weight-experiment helper."""

import pytest

from experiments.weight_experiment import BASELINE, EXPERIMENT, weights
from src import recommender
from src.main import DATA_PATH, DEFAULT_PROFILE, PROFILES, main
from src.recommender import load_songs, recommend_songs, score_song


@pytest.fixture(scope="module")
def catalog():
    return load_songs(str(DATA_PATH))


def test_profiles_are_the_four_phase4_profiles():
    assert list(PROFILES) == ["High-Energy Pop", "Chill Lofi", "Intense Rock", "Adversarial Listener"]
    assert PROFILES["Adversarial Listener"] == {
        "favorite_genre": "lofi", "favorite_mood": "sad", "target_energy": 0.90, "likes_acoustic": False,
    }


def test_default_run_shows_only_the_default_profile(capsys):
    main([])
    out = capsys.readouterr().out
    assert DEFAULT_PROFILE == "High-Energy Pop"
    assert "Profile:" not in out
    assert out.count("Because:") == 5
    assert "1. Sunrise City by Neon Echo" in out


def test_all_profiles_run_prints_five_results_per_profile(capsys, catalog):
    main(["--all-profiles"])
    out = capsys.readouterr().out
    assert out.count("Because:") == 5 * len(PROFILES)
    positions = [out.index(f"Profile: {name}") for name in PROFILES]
    assert positions == sorted(positions)
    for name, prefs in PROFILES.items():
        top_song = recommend_songs(prefs, catalog, k=1)[0][0]
        section = out[out.index(f"Profile: {name}"):]
        assert f"1. {top_song['title']} by {top_song['artist']}" in section


def test_experiment_weights_reach_score_song_and_are_restored(catalog):
    sunrise = catalog[0]
    pop = PROFILES["High-Energy Pop"]
    baseline_score, _ = score_song(pop, sunrise)
    with weights(EXPERIMENT) as max_score:
        experiment_score, _ = score_song(pop, sunrise)
        assert max_score == 12.0
    assert baseline_score == pytest.approx(10.76)
    assert experiment_score == pytest.approx(11.70)
    assert {name: getattr(recommender, name) for name in BASELINE} == BASELINE
    assert recommender.MAX_SCORE == 11.0


def test_experiment_weights_are_restored_after_an_error():
    with pytest.raises(RuntimeError):
        with weights(EXPERIMENT):
            raise RuntimeError("boom")
    assert {name: getattr(recommender, name) for name in BASELINE} == BASELINE
