"""Tests for the optional diversity-aware reranker (bonus challenge 3)."""

import copy
import random
from unittest.mock import patch

import pytest

from src import recommender
from src.main import DATA_PATH, PROFILES, main
from src.recommender import (
    ARTIST_REPEAT_PENALTY,
    GENRE_REPEAT_PENALTY,
    STRATEGIES,
    diversity_rerank,
    load_songs,
    recommend_diverse_songs,
    recommend_songs,
    score_song,
)

LOFI = PROFILES["Chill Lofi"]
ROCK = PROFILES["Intense Rock"]
ADVERSARIAL = PROFILES["Adversarial Listener"]


@pytest.fixture(scope="module")
def catalog():
    return load_songs(str(DATA_PATH))


def song(id, genre, artist, title=None):
    return {"id": id, "title": title or f"Song {id}", "artist": artist, "genre": genre, "mood": "happy"}


def scored(*entries):
    """Build (song, score, explanation) tuples the way recommend_songs does."""
    return [(s, score, f"reasons for {s['title']}") for s, score in entries]


def ids(picks):
    return [p.song["id"] for p in picks]


def titles(picks):
    return [p.song["title"] for p in picks]


def test_penalty_constants():
    assert GENRE_REPEAT_PENALTY == 1.5
    assert ARTIST_REPEAT_PENALTY == 2.0


# ---------------------------------------------------------------- the rule itself (synthetic songs)

def test_repeated_genre_is_penalized_per_earlier_pick():
    picks = diversity_rerank(scored(
        (song(1, "pop", "A"), 10.0),
        (song(2, "pop", "B"), 9.0),
        (song(3, "pop", "C"), 8.0),
        (song(4, "jazz", "D"), 7.0),
    ), k=4)
    # pop 10 -> pop 9 - 1.5 = 7.5 -> jazz 7.0 beats pop 8 - 3.0 = 5.0 -> pop 5.0
    assert ids(picks) == [1, 2, 4, 3]
    assert [p.genre_repeats for p in picks] == [0, 1, 0, 2]
    assert [p.genre_penalty for p in picks] == [0.0, 1.5, 0.0, 3.0]
    assert picks[3].adjusted_score == pytest.approx(5.0)


def test_repeated_artist_is_penalized_once_and_labels_are_normalized():
    picks = diversity_rerank(scored(
        (song(1, "rock", "Same Band"), 10.0),
        (song(2, "jazz", "  same   BAND "), 9.0),
        (song(3, "folk", "Other"), 8.0),
        (song(4, "r&b", "SAME band"), 7.5),
    ), k=4)
    # Same Band 10 -> Other 8 beats same band 9 - 2 = 7 -> 7.0 vs 7.5 - 2 = 5.5
    assert ids(picks) == [1, 3, 2, 4]
    assert [p.artist_repeat for p in picks] == [False, False, True, True]
    assert [p.artist_penalty for p in picks] == [0.0, 0.0, 2.0, 2.0]  # flat, not per repeat
    assert all(p.genre_penalty == 0.0 for p in picks)


def test_genre_labels_are_normalized():
    picks = diversity_rerank(scored((song(1, "Indie Pop", "A"), 10.0), (song(2, " indie  pop", "B"), 9.0)), k=2)
    assert picks[1].genre_penalty == 1.5


def test_no_repeats_means_no_change():
    entries = scored(*[(song(i, f"genre{i}", f"artist{i}"), 10.0 - i) for i in range(1, 7)])
    picks = diversity_rerank(entries, k=6)
    assert ids(picks) == [1, 2, 3, 4, 5, 6]
    assert all(p.adjusted_score == p.base_score for p in picks)
    assert all(p.penalty_explanation().startswith("No repetition penalty") for p in picks)


def test_penalties_only_count_selected_songs():
    # Two other jazz songs exist in the catalog but are never selected (k=2): no penalty.
    picks = diversity_rerank(scored(
        (song(1, "rock", "A"), 10.0),
        (song(2, "jazz", "B"), 9.0),
        (song(3, "jazz", "C"), 1.0),
        (song(4, "jazz", "D"), 0.5),
    ), k=2)
    assert ids(picks) == [1, 2]
    assert picks[1].genre_penalty == 0.0


def test_exact_ties_break_by_lower_id_even_after_penalties():
    picks = diversity_rerank(scored(
        (song(9, "pop", "A"), 10.0),
        (song(7, "pop", "B"), 7.3),   # 7.3 - 1.5 is 5.8 up to float noise
        (song(5, "jazz", "C"), 5.8),
    ), k=3)
    assert ids(picks) == [9, 5, 7]
    picks = diversity_rerank(scored((song(8, "rock", "A"), 4.0), (song(3, "folk", "B"), 4.0)), k=2)
    assert ids(picks) == [3, 8]


def test_zero_penalties_reproduce_the_plain_ranking(catalog):
    for mode in STRATEGIES:
        for profile in PROFILES.values():
            plain = recommend_songs(profile, catalog, k=18, mode=mode)
            reranked = diversity_rerank(plain, k=18, genre_penalty=0, artist_penalty=0)
            assert [(p.song["id"], p.base_score) for p in reranked] == [(s["id"], sc) for s, sc, _ in plain]


def test_penalty_constants_are_read_at_call_time(catalog):
    with patch.multiple(recommender, GENRE_REPEAT_PENALTY=0.0, ARTIST_REPEAT_PENALTY=0.0):
        assert ids(recommend_diverse_songs(LOFI, catalog)) == [s["id"] for s, _, _ in recommend_songs(LOFI, catalog)]
    assert ids(recommend_diverse_songs(LOFI, catalog)) != [s["id"] for s, _, _ in recommend_songs(LOFI, catalog)]


# ---------------------------------------------------------------- edge cases and validation

def test_k_and_empty_catalog_edge_cases(catalog):
    assert recommend_diverse_songs(LOFI, [], k=5) == []
    assert recommend_diverse_songs(LOFI, catalog, k=0) == []
    everything = recommend_diverse_songs(LOFI, catalog, k=100)
    assert sorted(ids(everything)) == sorted(s["id"] for s in catalog)
    with pytest.raises(ValueError):
        recommend_diverse_songs(LOFI, catalog, k=-1)
    with pytest.raises(TypeError):
        recommend_diverse_songs(LOFI, catalog, k="5")


def test_invalid_inputs_are_rejected(catalog):
    with pytest.raises(ValueError, match="unknown mode"):
        recommend_diverse_songs(LOFI, catalog, mode="most-diverse")
    with pytest.raises(ValueError, match="missing required field"):
        recommend_diverse_songs({"favorite_genre": "lofi"}, catalog)
    with pytest.raises(ValueError, match="genre_penalty"):
        diversity_rerank([], k=1, genre_penalty=-1.0)
    with pytest.raises(ValueError, match="artist"):
        diversity_rerank(scored(({"id": 1, "genre": "pop", "artist": None, "title": "x"}, 1.0)), k=1)


# ---------------------------------------------------------------- real catalog behavior

def test_real_catalog_before_and_after(catalog):
    assert titles(recommend_diverse_songs(LOFI, catalog)) == [
        "Library Rain", "Midnight Coding", "Spacewalk Thoughts", "Coffee Shop Stories", "Paper Boats",
    ]
    assert titles(recommend_diverse_songs(ADVERSARIAL, catalog)) == [
        "Midnight Coding", "Paper Boats", "Library Rain", "Storm Runner", "Gym Hero",
    ]
    # Rock is unchanged: Static Postcards drops to 6.86 - 3.00 = 3.86 but still beats Concrete Crown (3.61).
    assert ids(recommend_diverse_songs(ROCK, catalog)) == [s["id"] for s, _, _ in recommend_songs(ROCK, catalog)]


def test_artist_penalty_is_what_removes_focus_flow(catalog):
    # Focus Flow and Midnight Coding are both by LoRoom.
    without_artist = diversity_rerank(recommend_songs(ADVERSARIAL, catalog, k=18), k=5, artist_penalty=0.0)
    assert titles(without_artist)[1] == "Focus Flow"
    assert "Focus Flow" not in titles(recommend_diverse_songs(ADVERSARIAL, catalog))
    without_artist = diversity_rerank(recommend_songs(LOFI, catalog, k=18), k=5, artist_penalty=0.0)
    assert "Focus Flow" in titles(without_artist)
    assert "Focus Flow" not in titles(recommend_diverse_songs(LOFI, catalog))


def test_diversity_never_lowers_unique_genres_or_artists(catalog):
    for mode in STRATEGIES:
        for profile in PROFILES.values():
            plain = [s for s, _, _ in recommend_songs(profile, catalog, mode=mode)]
            diverse = [p.song for p in recommend_diverse_songs(profile, catalog, mode=mode)]
            assert len({s["genre"] for s in diverse}) >= len({s["genre"] for s in plain})
            assert len({s["artist"] for s in diverse}) >= len({s["artist"] for s in plain})


# ---------------------------------------------------------------- modes, explanations, purity

@pytest.mark.parametrize("mode", list(STRATEGIES))
def test_works_with_every_mode_and_keeps_base_scores(catalog, mode):
    for profile in PROFILES.values():
        picks = recommend_diverse_songs(profile, catalog, mode=mode)
        assert len(picks) == 5
        assert picks[0].song["id"] == recommend_songs(profile, catalog, k=1, mode=mode)[0][0]["id"]
        for p in picks:
            base, reasons = score_song(profile, p.song, mode=mode)
            assert p.base_score == base                 # relevance is never modified
            assert p.explanation == "; ".join(reasons)  # same reasons as the plain ranking
            assert p.adjusted_score == pytest.approx(p.base_score - p.genre_penalty - p.artist_penalty)


def test_penalty_explanation_states_every_part(catalog):
    static = recommend_diverse_songs(ROCK, catalog)[3]
    assert static.song["title"] == "Static Postcards"
    assert static.penalty_explanation() == (
        "Genre rock already picked 2x (-3.00); adjusted 3.86 = relevance 6.86 minus penalties"
    )
    both = diversity_rerank(scored((song(1, "pop", "X"), 10.0), (song(2, "pop", "X"), 9.0)), k=2)[1]
    assert both.penalty_explanation() == (
        "Genre pop already picked 1x (-1.50); Artist X already picked (-2.00); "
        "adjusted 5.50 = relevance 9.00 minus penalties"
    )


def test_inputs_are_not_mutated(catalog):
    songs = copy.deepcopy(catalog)
    entries = recommend_songs(LOFI, songs, k=18)
    entries_before = list(entries)
    recommend_diverse_songs(LOFI, songs)
    diversity_rerank(entries, k=5)
    assert songs == catalog
    assert entries == entries_before


def test_deterministic_and_independent_of_catalog_order(catalog):
    for mode in STRATEGIES:
        for profile in PROFILES.values():
            expected = ids(recommend_diverse_songs(profile, catalog, k=18, mode=mode))
            shuffled = list(catalog)
            random.Random(11).shuffle(shuffled)
            assert ids(recommend_diverse_songs(profile, shuffled, k=18, mode=mode)) == expected
            assert ids(recommend_diverse_songs(profile, catalog, k=18, mode=mode)) == expected


# ---------------------------------------------------------------- backward compatibility and CLI

def test_plain_ranking_is_unchanged(catalog):
    assert [s["id"] for s, _, _ in recommend_songs(LOFI, catalog)] == [4, 2, 9, 6, 7]
    assert [s["id"] for s, _, _ in recommend_songs(ROCK, catalog)] == [3, 16, 5, 17, 12]


def test_cli_default_has_no_diversity_output(capsys):
    main([])
    out = capsys.readouterr().out
    assert "Diversity" not in out and "Relevance score" not in out


def test_cli_diverse_labels_relevance_and_adjusted_scores(capsys):
    main(["--all-profiles", "--diverse"])
    out = capsys.readouterr().out
    assert out.count("Diversity reranking: on") == 1
    assert out.count("Relevance score:") == 20 and out.count("Adjusted for diversity:") == 20
    assert "Genre rock already picked 2x (-3.00)" in out
    assert "Score:" not in out  # an adjusted score is never shown as a plain "Score"


def test_cli_diverse_combines_with_mode(capsys):
    main(["--mode", "energy-focused", "--diverse"])
    out = capsys.readouterr().out
    assert "Ranking mode: energy-focused" in out and "Diversity reranking: on" in out
    assert out.count("Relevance score:") == 5
