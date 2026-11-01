"""Tests for the five advanced song attributes and optional advanced scoring (bonus challenge 1)."""

import csv
import re

import pytest

from src.main import ADVANCED_PROFILE_PREFS, DATA_PATH, PROFILES, main, profile_prefs
from src.recommender import (
    ADVANCED_COLUMNS,
    ADVANCED_MAX_SCORE,
    REQUIRED_COLUMNS,
    STRATEGIES,
    Recommender,
    Song,
    UserProfile,
    load_songs,
    max_possible_score,
    recommend_diverse_songs,
    recommend_songs,
    score_song,
)
from src.table import diverse_table, recommendations_table

BASE_ROW = ["1", "Song", "Artist", "pop", "happy", "0.5", "100", "0.5", "0.5", "0.5"]
ADV_ROW = ["60", "2010", "0.30", "0.90", "dreamy|focused"]
ADV_PREFS = {
    "favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.5, "likes_acoustic": False,
    "target_popularity": 80, "preferred_decade": 2010, "target_instrumentalness": 0.5,
    "target_vocal_intensity": 0.6, "preferred_tags": ["focused", "euphoric"],
}


@pytest.fixture(scope="module")
def catalog():
    return load_songs(str(DATA_PATH))


def write_csv(tmp_path, rows, header):
    path = tmp_path / "songs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def advanced_csv(tmp_path, column=None, value=None):
    row = BASE_ROW + ADV_ROW
    if column:
        row[(REQUIRED_COLUMNS + ADVANCED_COLUMNS).index(column)] = value
    return write_csv(tmp_path, [row], REQUIRED_COLUMNS + ADVANCED_COLUMNS)


def points(reasons):
    return [float(p) for p in re.findall(r"\(\+(\d+\.\d\d)\)", "; ".join(reasons))]


def by_title(catalog, title):
    return next(song for song in catalog if song["title"] == title)


def titles(results):
    return [song["title"] for song, _, _ in results]


# ---------------------------------------------------------------- dataset and parsing

def test_catalog_has_all_five_attributes_for_all_18_songs(catalog):
    assert len(catalog) == 18
    for song in catalog:
        assert isinstance(song["popularity"], int) and 0 <= song["popularity"] <= 100
        assert isinstance(song["release_decade"], int) and song["release_decade"] % 10 == 0
        assert 0.0 <= song["instrumentalness"] <= 1.0
        assert 0.0 <= song["vocal_intensity"] <= 1.0
        assert isinstance(song["mood_tags"], tuple) and song["mood_tags"]
        assert len(set(song["mood_tags"])) == len(song["mood_tags"])
    sunrise = catalog[0]
    assert (sunrise["popularity"], sunrise["release_decade"], sunrise["instrumentalness"],
            sunrise["vocal_intensity"], sunrise["mood_tags"]) == (78, 2020, 0.05, 0.70, ("uplifting", "energetic"))


def test_csv_header_has_exactly_five_new_columns():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == REQUIRED_COLUMNS + ADVANCED_COLUMNS


def test_original_schema_file_still_loads_without_advanced_values(tmp_path):
    songs = load_songs(write_csv(tmp_path, [BASE_ROW], REQUIRED_COLUMNS))
    assert set(songs[0]) == set(REQUIRED_COLUMNS)


def test_partial_advanced_columns_are_rejected(tmp_path):
    path = write_csv(tmp_path, [BASE_ROW + ["60"]], REQUIRED_COLUMNS + ["popularity"])
    with pytest.raises(ValueError, match="some advanced columns but is missing: release_decade"):
        load_songs(path)


def test_mood_tags_are_normalized(tmp_path):
    songs = load_songs(advanced_csv(tmp_path, "mood_tags", " Dreamy | FOCUSED "))
    assert songs[0]["mood_tags"] == ("dreamy", "focused")


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("popularity", "101", "popularity must be between 0 and 100"),
        ("popularity", "-1", "popularity must be between 0 and 100"),
        ("popularity", "78.5", "popularity must be a whole number"),
        ("popularity", "", "missing value for popularity"),
        ("release_decade", "2015", "release_decade must be a decade"),
        ("release_decade", "1890", "release_decade must be a decade"),
        ("release_decade", "2030", "release_decade must be a decade"),
        ("release_decade", "nineties", "release_decade must be a whole number"),
        ("instrumentalness", "1.2", "instrumentalness must be between 0 and 1"),
        ("instrumentalness", "-0.1", "instrumentalness must be between 0 and 1"),
        ("instrumentalness", "lots", "instrumentalness must be a number"),
        ("vocal_intensity", "2", "vocal_intensity must be between 0 and 1"),
        ("vocal_intensity", "nan", "vocal_intensity must be a finite number"),
        ("mood_tags", "happy", "mood_tags: unknown mood tag 'happy'"),
        ("mood_tags", "dreamy|dreamy", "mood_tags has a repeated tag"),
        ("mood_tags", "dreamy||focused", "mood_tags: mood tags must be non-empty"),
        ("mood_tags", " ", "missing value for mood_tags"),
    ],
)
def test_invalid_advanced_values_are_rejected(tmp_path, column, value, message):
    with pytest.raises(ValueError, match=f"line 2: {message}"):
        load_songs(advanced_csv(tmp_path, column, value))


def test_valid_edge_values_load(tmp_path):
    for column, value in [("popularity", "0"), ("popularity", "100"), ("release_decade", "1900"),
                          ("release_decade", "2020"), ("instrumentalness", "0"), ("vocal_intensity", "1")]:
        assert load_songs(advanced_csv(tmp_path, column, value))


# ---------------------------------------------------------------- the five contributions

def test_all_five_contributions_and_their_reasons():
    song = {"id": 1, "title": "T", "genre": "rock", "mood": "sad", "energy": 0.5, "acousticness": 0.5,
            "popularity": 60, "release_decade": 2010, "instrumentalness": 0.3, "vocal_intensity": 0.9,
            "mood_tags": ("dreamy", "focused")}
    basic, basic_reasons = score_song(ADV_PREFS, song)
    score, reasons = score_song(ADV_PREFS, song, advanced=True)
    assert reasons[:4] == basic_reasons
    assert reasons[4:] == [
        "Popularity 60 vs target 80 (+0.40)",          # 0.5 * (1 - 20/100)
        "Decade match: 2010s (+0.50)",
        "Instrumentalness 0.30 vs target 0.50 (+0.40)",  # 0.5 * (1 - 0.2)
        "Vocal intensity 0.90 vs target 0.60 (+0.35)",   # 0.5 * (1 - 0.3)
        "Mood tags 1/2 matched: focused (+0.50)",        # 1 of 2 unique preferred tags
    ]
    assert score - basic == pytest.approx(0.40 + 0.50 + 0.40 + 0.35 + 0.50)


def test_decade_mismatch_and_no_tag_match_score_zero():
    song = {"id": 1, "genre": "pop", "mood": "happy", "energy": 0.5, "acousticness": 0.5,
            "popularity": 80, "release_decade": 1990, "instrumentalness": 0.5, "vocal_intensity": 0.6,
            "mood_tags": ("melancholic",)}
    _, reasons = score_song(ADV_PREFS, song, advanced=True)
    assert "Decade 1990s is not 2010s (+0.00)" in reasons
    assert "Mood tags 0/2 matched (+0.00)" in reasons


def test_each_preferred_tag_counts_once_and_sets_are_deterministic():
    song = {"id": 1, "genre": "pop", "mood": "happy", "energy": 0.5, "acousticness": 0.5,
            "popularity": 80, "release_decade": 2010, "instrumentalness": 0.5, "vocal_intensity": 0.6,
            "mood_tags": ("focused", "dreamy")}
    duplicated = {**ADV_PREFS, "preferred_tags": ["Focused", "focused", " FOCUSED ", "euphoric"]}
    assert score_song(duplicated, song, advanced=True)[1][-1] == "Mood tags 1/2 matched: focused (+0.50)"
    as_set = {**ADV_PREFS, "preferred_tags": {"focused", "dreamy"}}
    assert score_song(as_set, song, advanced=True)[1][-1] == "Mood tags 2/2 matched: dreamy, focused (+1.00)"


def test_maximum_advanced_score():
    assert ADVANCED_MAX_SCORE == 3.0
    perfect = {"id": 1, "genre": "pop", "mood": "happy", "energy": 0.5, "acousticness": 0.0,
               "popularity": 80, "release_decade": 2010, "instrumentalness": 0.5, "vocal_intensity": 0.6,
               "mood_tags": ("focused", "euphoric", "dreamy")}
    score, _ = score_song(ADV_PREFS, perfect, advanced=True)
    assert score == pytest.approx(14.0) == max_possible_score("baseline", advanced=True)
    assert max_possible_score() == 11.0
    assert max_possible_score("mood-first", advanced=True) == 13.0
    assert max_possible_score("energy-focused", advanced=True) == 15.0


# ---------------------------------------------------------------- validation

def test_missing_advanced_preferences_are_reported(catalog):
    basic = PROFILES["High-Energy Pop"]
    with pytest.raises(ValueError, match="missing: target_popularity, preferred_decade"):
        recommend_songs(basic, catalog, advanced=True)
    with pytest.raises(ValueError, match="missing: preferred_tags"):
        score_song({**ADV_PREFS, "preferred_tags": None}, catalog[0], advanced=True)
    with pytest.raises(ValueError, match="missing"):
        recommend_songs(basic, [], advanced=True)  # checked even with an empty catalog


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("target_popularity", 120, "target_popularity must be a number between 0 and 100"),
        ("target_popularity", "80", "target_popularity must be a number"),
        ("preferred_decade", 2015, "preferred_decade must be a decade"),
        ("preferred_decade", 2010.0, "preferred_decade must be a whole number"),
        ("preferred_decade", True, "preferred_decade must be a whole number"),
        ("target_instrumentalness", 1.5, "target_instrumentalness must be a number between 0 and 1"),
        ("target_vocal_intensity", -0.2, "target_vocal_intensity must be a number between 0 and 1"),
        ("preferred_tags", "dreamy", "preferred_tags must be a non-empty list"),
        ("preferred_tags", [], "preferred_tags must be a non-empty list"),
        ("preferred_tags", ["dreamy", "happy"], "unknown mood tag 'happy'"),
    ],
)
def test_invalid_advanced_preferences_are_rejected(catalog, key, value, message):
    with pytest.raises(ValueError, match=message):
        score_song({**ADV_PREFS, key: value}, catalog[0], advanced=True)


def test_songs_without_advanced_data_are_never_scored_as_if_they_had_it(tmp_path):
    songs = load_songs(write_csv(tmp_path, [BASE_ROW], REQUIRED_COLUMNS))
    assert recommend_songs(ADV_PREFS, songs)  # fine without advanced scoring
    with pytest.raises(ValueError, match="has no popularity, release_decade.*advanced columns"):
        recommend_songs(ADV_PREFS, songs, advanced=True)
    old_style = Song(1, "T", "A", "pop", "happy", 0.5, 100.0, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="has no popularity"):
        Recommender([old_style], advanced=True).recommend(UserProfile(**ADV_PREFS))


def test_advanced_flag_must_be_a_bool(catalog):
    with pytest.raises(TypeError):
        recommend_songs(ADV_PREFS, catalog, advanced="yes")
    with pytest.raises(TypeError):
        Recommender([], advanced=1)


# ---------------------------------------------------------------- backward compatibility

def test_basic_scoring_ignores_advanced_preferences_and_attributes(catalog):
    for name in PROFILES:
        plain = recommend_songs(PROFILES[name], catalog, k=18)
        with_extras = recommend_songs(profile_prefs(name, advanced=True), catalog, k=18)
        assert [(s["id"], sc, why) for s, sc, why in plain] == [(s["id"], sc, why) for s, sc, why in with_extras]
    assert [s["id"] for s, _, _ in recommend_songs(PROFILES["Chill Lofi"], catalog)] == [4, 2, 9, 6, 7]


def test_demo_profiles_are_unchanged_and_advanced_prefs_are_separate():
    assert all(set(p) == {"favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"}
               for p in PROFILES.values())
    assert list(ADVANCED_PROFILE_PREFS) == list(PROFILES)


def test_original_dataclass_construction_still_works(catalog):
    song = Song(1, "T", "A", "pop", "happy", 0.8, 120.0, 0.8, 0.8, 0.2)
    user = UserProfile("pop", "happy", 0.8, False)
    assert Recommender([song]).recommend(user) == [song]
    assert song.popularity is None and user.preferred_tags is None


def test_oop_interface_supports_advanced_scoring(catalog):
    songs = [Song(**song) for song in catalog]
    prefs = profile_prefs("Intense Rock", advanced=True)
    user = UserProfile(**prefs)
    rec = Recommender(songs, advanced=True)
    expected = [s["id"] for s, _, _ in recommend_songs(prefs, catalog, advanced=True)]
    assert [song.id for song in rec.recommend(user)] == expected
    _, reasons = score_song(prefs, catalog[2], advanced=True)
    assert rec.explain_recommendation(user, songs[2]) == "; ".join(reasons)


# ---------------------------------------------------------------- real catalog effect

def test_advanced_scoring_changes_real_rankings(catalog):
    pop = profile_prefs("High-Energy Pop", advanced=True)
    assert titles(recommend_songs(PROFILES["High-Energy Pop"], catalog))[1:3] == ["Gym Hero", "Golden Hour Lagos"]
    assert titles(recommend_songs(pop, catalog, advanced=True))[1:3] == ["Golden Hour Lagos", "Gym Hero"]
    rock = profile_prefs("Intense Rock", advanced=True)
    assert titles(recommend_songs(rock, catalog, advanced=True))[0] == "Breakwater"
    adversarial = profile_prefs("Adversarial Listener", advanced=True)
    top = titles(recommend_songs(adversarial, catalog, advanced=True))
    assert "Nocturne in Grey" in top and "Storm Runner" not in top


@pytest.mark.parametrize("mode", list(STRATEGIES))
def test_advanced_points_are_the_same_in_every_mode(catalog, mode):
    for name in PROFILES:
        prefs = profile_prefs(name, advanced=True)
        for song in catalog:
            basic, _ = score_song(prefs, song, mode=mode)
            advanced, reasons = score_song(prefs, song, mode=mode, advanced=True)
            baseline_basic, _ = score_song(prefs, song)
            baseline_advanced, _ = score_song(prefs, song, advanced=True)
            assert advanced - basic == pytest.approx(baseline_advanced - baseline_basic)
            assert len(points(reasons)) == 9
            assert sum(points(reasons)) == pytest.approx(advanced, abs=0.05)
            assert advanced <= max_possible_score(mode, advanced=True)


def test_diversity_works_with_advanced_scoring(catalog):
    prefs = profile_prefs("Intense Rock", advanced=True)
    picks = recommend_diverse_songs(prefs, catalog, advanced=True)
    for pick in picks:
        assert pick.base_score == score_song(prefs, pick.song, advanced=True)[0]
        assert "Mood tags" in pick.explanation
    assert [p.song["title"] for p in picks] == [
        "Breakwater", "Storm Runner", "Gym Hero", "Concrete Crown", "Golden Hour Lagos",
    ]


def test_tables_work_with_advanced_scoring(catalog):
    prefs = profile_prefs("High-Energy Pop", advanced=True)
    max_score = max_possible_score(advanced=True)
    text = recommendations_table(recommend_songs(prefs, catalog, advanced=True), max_score)
    assert "(max 14)" in text and "13.23" in text and "Mood tags 2/2 matched:" in text
    text = diverse_table(recommend_diverse_songs(prefs, catalog, advanced=True), max_score)
    assert "(max 14)" in text and "-> Adjusted" in text and "Decade match: 2020s" in text


# ---------------------------------------------------------------- CLI

def run(capsys, *args):
    main(list(args))
    return capsys.readouterr().out


def test_cli_default_has_no_advanced_output(capsys):
    out = run(capsys)
    assert "Advanced" not in out and "popularity" not in out and "Mood tags" not in out


@pytest.mark.parametrize(
    "args, max_score, profiles",
    [
        (["--advanced"], 14, 1),
        (["--all-profiles", "--advanced"], 14, 4),
        (["--mode", "mood-first", "--advanced"], 13, 1),
        (["--advanced", "--diverse", "--table"], 14, 1),
        (["--all-profiles", "--mode", "energy-focused", "--advanced", "--diverse", "--table"], 15, 4),
    ],
)
def test_cli_advanced_combinations(capsys, args, max_score, profiles):
    out = run(capsys, *args)
    assert out.count("Advanced scoring: on") == 1
    assert f"max {max_score}" in out
    assert out.count("preferred_tags:") == profiles
    assert " ".join(out.split()).count("Mood tags") == 5 * profiles
    if "--table" in args:
        assert f"(max {max_score})" in out
