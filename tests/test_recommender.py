import copy
import csv
import random
from dataclasses import replace
from pathlib import Path

import pytest

from src.recommender import (
    MAX_SCORE,
    REQUIRED_COLUMNS,
    Recommender,
    Song,
    UserProfile,
    load_songs,
    recommend_songs,
    score_song,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"

LOFI_PROFILE = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.40,
    "likes_acoustic": True,
}
ROCK_PROFILE = {
    "favorite_genre": "rock",
    "favorite_mood": "intense",
    "target_energy": 0.90,
    "likes_acoustic": False,
}


def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---------------------------------------------------------------- helpers

@pytest.fixture(scope="module")
def catalog():
    return load_songs(str(DATA_PATH))


def song_by_title(songs, title):
    return next(song for song in songs if song["title"] == title)


def make_song(**overrides):
    song = {
        "id": 1, "title": "T", "artist": "A", "genre": "pop", "mood": "happy",
        "energy": 0.5, "tempo_bpm": 100.0, "valence": 0.5, "danceability": 0.5,
        "acousticness": 0.5,
    }
    song.update(overrides)
    return song


def write_csv(tmp_path, rows, header=None):
    path = tmp_path / "songs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header or REQUIRED_COLUMNS)
        writer.writerows(rows)
    return str(path)


VALID_ROW = ["1", "Song", "Artist", "pop", "happy", "0.5", "100", "0.5", "0.5", "0.5"]


def row_with(column, value):
    row = list(VALID_ROW)
    row[REQUIRED_COLUMNS.index(column)] = value
    return row


# ---------------------------------------------------------------- CSV loading

def test_load_songs_reads_all_18_rows(catalog):
    assert len(catalog) == 18
    assert [song["id"] for song in catalog] == list(range(1, 19))


def test_load_songs_converts_types(catalog):
    song = catalog[0]
    assert isinstance(song["id"], int)
    for col in ["energy", "tempo_bpm", "valence", "danceability", "acousticness"]:
        assert isinstance(song[col], float)
    for col in ["title", "artist", "genre", "mood"]:
        assert isinstance(song[col], str)
    # Bonus challenge 1 added five advanced columns; the original ten values are unchanged.
    assert {col: song[col] for col in REQUIRED_COLUMNS} == {
        "id": 1, "title": "Sunrise City", "artist": "Neon Echo", "genre": "pop",
        "mood": "happy", "energy": 0.82, "tempo_bpm": 118.0, "valence": 0.84,
        "danceability": 0.79, "acousticness": 0.18,
    }
    assert set(song) == set(REQUIRED_COLUMNS) | {
        "popularity", "release_decade", "instrumentalness", "vocal_intensity", "mood_tags",
    }


def test_load_songs_rejects_duplicate_ids(tmp_path):
    path = write_csv(tmp_path, [VALID_ROW, VALID_ROW])
    with pytest.raises(ValueError, match="line 3: duplicate id 1"):
        load_songs(path)


def test_load_songs_rejects_missing_column(tmp_path):
    header = [col for col in REQUIRED_COLUMNS if col != "mood"]
    row = [v for col, v in zip(REQUIRED_COLUMNS, VALID_ROW) if col != "mood"]
    path = write_csv(tmp_path, [row], header=header)
    with pytest.raises(ValueError, match="missing required column.*mood"):
        load_songs(path)


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("id", "abc", "id must be an integer"),
        ("genre", "", "missing value for genre"),
        ("energy", "loud", "energy must be a number"),
        ("energy", "1.5", "energy must be between 0 and 1"),
        ("acousticness", "-0.1", "acousticness must be between 0 and 1"),
        ("valence", "nan", "valence must be a finite number"),
        ("danceability", "inf", "danceability must be a finite number"),
        ("tempo_bpm", "0", "tempo_bpm must be positive"),
    ],
)
def test_load_songs_rejects_malformed_values(tmp_path, column, value, message):
    path = write_csv(tmp_path, [row_with(column, value)])
    with pytest.raises(ValueError, match=f"line 2: {message}"):
        load_songs(path)


def test_load_songs_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Song catalog not found"):
        load_songs(str(tmp_path / "nope.csv"))


# ---------------------------------------------------------------- scoring

def test_genre_and_mood_matches_add_their_weights():
    prefs = {**ROCK_PROFILE, "target_energy": 0.5}
    song = make_song(genre="rock", mood="calm", energy=0.5, acousticness=1.0)
    score, _ = score_song(prefs, song)
    assert score == pytest.approx(4.0 + 0.0 + 3.0 + 0.0)

    score, _ = score_song(prefs, make_song(genre="jazz", mood="intense", energy=0.5, acousticness=1.0))
    assert score == pytest.approx(0.0 + 3.0 + 3.0 + 0.0)


def test_labels_are_normalized_for_case_and_whitespace():
    prefs = {**LOFI_PROFILE, "favorite_genre": "  Indie   POP ", "favorite_mood": "CHILL"}
    score, reasons = score_song(prefs, make_song(genre="indie pop", mood=" chill"))
    assert reasons[0] == "Genre match: indie pop (+4.00)"
    assert reasons[1] == "Mood match:  chill (+3.00)"
    assert score >= 7.0


def test_energy_rewards_closeness_not_intensity():
    prefs = {**LOFI_PROFILE, "target_energy": 0.40}
    scores = [score_song(prefs, make_song(energy=e))[0] for e in [0.40, 0.45, 0.60, 0.90]]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[1] > scores[2] > scores[3]
    # Higher energy than the target is not rewarded over an equally distant lower energy.
    below, _ = score_song(prefs, make_song(energy=0.30))
    above, _ = score_song(prefs, make_song(energy=0.50))
    assert below == pytest.approx(above)


@pytest.mark.parametrize("likes_acoustic, expected", [(True, 0.8), (False, 0.2)])
def test_acoustic_preference(likes_acoustic, expected):
    prefs = {"favorite_genre": "x", "favorite_mood": "x", "target_energy": 0.0,
             "likes_acoustic": likes_acoustic}
    song = make_song(energy=1.0, acousticness=0.8)  # energy contributes 0
    score, _ = score_song(prefs, song)
    assert score == pytest.approx(expected)


def test_maximum_score_is_eleven():
    assert MAX_SCORE == 11.0
    score, _ = score_song(LOFI_PROFILE, make_song(genre="lofi", mood="chill", energy=0.40, acousticness=1.0))
    assert score == pytest.approx(11.0)


@pytest.mark.parametrize(
    "bad_prefs",
    [
        {**LOFI_PROFILE, "likes_acoustic": "False"},
        {**LOFI_PROFILE, "likes_acoustic": 0},
        {**LOFI_PROFILE, "target_energy": 1.5},
        {**LOFI_PROFILE, "target_energy": "0.4"},
        {**LOFI_PROFILE, "target_energy": True},
        {**LOFI_PROFILE, "favorite_genre": 7},
        {**LOFI_PROFILE, "favorite_mood": "   "},
        {key: v for key, v in LOFI_PROFILE.items() if key != "likes_acoustic"},
        {"genre": "pop", "mood": "happy", "energy": 0.8},  # old starter keys
    ],
)
def test_invalid_profiles_are_rejected(bad_prefs):
    with pytest.raises(ValueError):
        score_song(bad_prefs, make_song())


@pytest.mark.parametrize(
    "bad_song",
    [make_song(energy=1.2), make_song(acousticness="0.5"), make_song(genre=None),
     {k: v for k, v in make_song().items() if k != "mood"}],
)
def test_malformed_songs_are_rejected(bad_song):
    with pytest.raises(ValueError):
        score_song(LOFI_PROFILE, bad_song)


def test_reasons_match_score_components():
    song = make_song(genre="rock", mood="moody", energy=0.62, acousticness=0.30)
    score, reasons = score_song(ROCK_PROFILE, song)
    assert reasons == [
        "Genre match: rock (+4.00)",
        "Mood moody is not intense (+0.00)",
        "Energy 0.62 vs target 0.90 (+2.16)",
        "Acousticness 0.30, you prefer less acoustic (+0.70)",
    ]
    assert score == pytest.approx(6.86)
    assert not any(reason.startswith("Mood match") for reason in reasons)


# ---------------------------------------------------------------- worked examples (Phase 2)

@pytest.mark.parametrize(
    "profile, title, expected",
    [
        (LOFI_PROFILE, "Library Rain", 10.71),
        (LOFI_PROFILE, "Midnight Coding", 10.65),
        (LOFI_PROFILE, "Spacewalk Thoughts", 6.56),
        (LOFI_PROFILE, "Storm Runner", 1.57),
        (ROCK_PROFILE, "Storm Runner", 10.87),
        (ROCK_PROFILE, "Gym Hero", 6.86),
        (ROCK_PROFILE, "Static Postcards", 6.86),
    ],
)
def test_worked_examples_from_csv(catalog, profile, title, expected):
    score, _ = score_song(profile, song_by_title(catalog, title))
    assert score == pytest.approx(expected)


# ---------------------------------------------------------------- ranking

def ranked_ids(results):
    return [song["id"] for song, _, _ in results]


def test_lofi_ranking_on_real_catalog(catalog):
    assert ranked_ids(recommend_songs(LOFI_PROFILE, catalog)) == [4, 2, 9, 6, 7]


def test_rock_ranking_breaks_tie_by_lower_id(catalog):
    results = recommend_songs(ROCK_PROFILE, catalog)
    assert ranked_ids(results) == [3, 16, 5, 17, 12]
    assert results[2][1] == pytest.approx(results[3][1])  # Gym Hero ties Static Postcards


def test_ranking_does_not_depend_on_catalog_order(catalog):
    expected = ranked_ids(recommend_songs(ROCK_PROFILE, catalog, k=18))
    shuffled = list(catalog)
    random.Random(42).shuffle(shuffled)
    assert ranked_ids(recommend_songs(ROCK_PROFILE, shuffled, k=18)) == expected
    assert ranked_ids(recommend_songs(ROCK_PROFILE, list(reversed(catalog)), k=18)) == expected


def test_equal_scores_put_lower_id_first():
    songs = [make_song(id=9), make_song(id=2), make_song(id=5)]
    assert ranked_ids(recommend_songs(LOFI_PROFILE, songs)) == [2, 5, 9]


def test_scores_are_descending_and_unrounded(catalog):
    results = recommend_songs(LOFI_PROFILE, catalog, k=18)
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)
    for song, score, explanation in results:
        expected_score, reasons = score_song(LOFI_PROFILE, song)
        assert score == expected_score
        assert explanation == "; ".join(reasons)


def test_k_edge_cases(catalog):
    assert len(recommend_songs(LOFI_PROFILE, catalog)) == 5
    assert recommend_songs(LOFI_PROFILE, catalog, k=0) == []
    assert len(recommend_songs(LOFI_PROFILE, catalog, k=100)) == 18
    assert recommend_songs(LOFI_PROFILE, [], k=5) == []
    with pytest.raises(ValueError):
        recommend_songs(LOFI_PROFILE, catalog, k=-1)
    for bad_k in ["3", 2.0, None, True]:
        with pytest.raises(TypeError):
            recommend_songs(LOFI_PROFILE, catalog, k=bad_k)


def test_invalid_profile_rejected_even_with_empty_catalog():
    with pytest.raises(ValueError):
        recommend_songs({"genre": "pop"}, [], k=5)


def test_recommend_does_not_mutate_input(catalog):
    songs = list(reversed(catalog))
    before = copy.deepcopy(songs)
    results = recommend_songs(LOFI_PROFILE, songs, k=3)
    assert songs == before
    assert results[0][0] is song_by_title(songs, "Library Rain")


# ---------------------------------------------------------------- OOP consistency

def test_oop_recommend_matches_functional_ranking(catalog):
    songs = [Song(**song) for song in reversed(catalog)]
    user = UserProfile(**ROCK_PROFILE)
    results = Recommender(songs).recommend(user, k=18)
    assert all(isinstance(song, Song) for song in results)
    assert [song.id for song in results] == ranked_ids(recommend_songs(ROCK_PROFILE, catalog, k=18))
    assert all(any(song is original for original in songs) for song in results)


def test_oop_explanation_uses_score_song_reasons(catalog):
    song = Song(**song_by_title(catalog, "Midnight Coding"))
    user = UserProfile(**LOFI_PROFILE)
    explanation = Recommender([song]).explain_recommendation(user, song)
    _, reasons = score_song(LOFI_PROFILE, song_by_title(catalog, "Midnight Coding"))
    assert explanation == "; ".join(reasons)
    assert "Energy 0.42 vs target 0.40 (+2.94)" in explanation


def test_oop_rejects_string_boolean():
    user = replace(UserProfile(**LOFI_PROFILE), likes_acoustic="False")
    with pytest.raises(ValueError):
        make_small_recommender().recommend(user)
