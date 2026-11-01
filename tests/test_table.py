"""Tests for the --table ASCII output (bonus challenge 4)."""

import copy

import pytest

from src.main import DATA_PATH, PROFILES, main
from src.recommender import (
    diversity_rerank,
    get_strategy,
    load_songs,
    recommend_diverse_songs,
    recommend_songs,
)
from src.table import REASON_WIDTH, diverse_table, recommendations_table, render_table, wrap_cell

POP = PROFILES["High-Energy Pop"]
ROCK = PROFILES["Intense Rock"]


@pytest.fixture(scope="module")
def catalog():
    return load_songs(str(DATA_PATH))


def parse_table(text):
    """Return [header, row, row, ...]; each is a list of columns, each column a list of lines."""
    rows, current = [], None
    for line in text.splitlines():
        if line.startswith("+"):
            if current:
                rows.append(current)
            current = None
            continue
        cells = [cell.strip() for cell in line[2:-2].split(" | ")]
        if current is None:
            current = [[] for _ in cells]
        for column, cell in zip(current, cells):
            if cell:
                column.append(cell)
    return rows


def column_text(lines):
    """Join a wrapped column back into one whitespace-normalized string."""
    return " ".join(" ".join(lines).split())


def max_score(mode):
    return get_strategy(mode).weights().max_score


# ---------------------------------------------------------------- the renderer

def test_render_table_has_headers_borders_and_equal_width_lines():
    text = render_table(["A", "B"], [["1", "x"], ["2", "y"]], [3, 5])
    lines = text.splitlines()
    assert lines[1] == "| A   | B     |"
    assert lines[2].startswith("+=====")
    assert len({len(line) for line in lines}) == 1


def test_long_text_wraps_without_losing_words():
    text = "Acousticness 0.18, you prefer less acoustic (+0.82) and many more words to force wrapping"
    lines = wrap_cell(text, 20)
    assert len(lines) > 1
    assert all(len(line) <= 20 for line in lines)
    assert lines[1].startswith("  ")  # continuation lines are indented
    assert " ".join(" ".join(lines).split()) == text
    table_lines = render_table(["Why"], [[text]], [20]).splitlines()
    assert len({len(line) for line in table_lines}) == 1


def test_words_longer_than_the_column_are_split_not_clipped():
    lines = wrap_cell("Supercalifragilistic", 8)
    assert "".join(line.strip() for line in lines) == "Supercalifragilistic"


def test_explicit_newlines_and_empty_cells():
    assert wrap_cell("one\ntwo", 10) == ["one", "two"]
    assert wrap_cell("", 10) == [""]


def test_mismatched_columns_are_rejected():
    with pytest.raises(ValueError):
        render_table(["A", "B"], [["only one"]], [3, 3])


# ---------------------------------------------------------------- real recommendations

@pytest.mark.parametrize("mode", ["baseline", "mood-first", "energy-focused"])
def test_table_shows_every_recommendation_score_and_reason(catalog, mode):
    for profile in PROFILES.values():
        recs = recommend_songs(profile, catalog, mode=mode)
        header, *rows = parse_table(recommendations_table(recs, max_score(mode)))
        assert column_text(header[0]) == "#"
        assert column_text(header[3]) == f"Score (max {max_score(mode):g})"
        assert column_text(header[4]) == "Why (score breakdown)"
        assert len(rows) == len(recs) == 5
        for rank, (row, (song, score, explanation)) in enumerate(zip(rows, recs), start=1):
            assert row[0] == [str(rank)]
            assert row[1] == [song["title"], f"by {song['artist']}"]
            assert row[2] == [song["genre"], song["mood"]]
            assert row[3] == [f"{score:.2f}"]
            reasons = column_text(row[4])
            for reason in explanation.split("; "):
                assert reason in reasons


def test_mode_specific_scores_and_reasons_are_shown(catalog):
    mood_first = recommendations_table(recommend_songs(POP, catalog, mode="mood-first"), 10.0)
    assert "(max 10)" in mood_first and "9.76" in mood_first and "Mood match: happy (+4.00)" in mood_first
    energy = recommendations_table(recommend_songs(POP, catalog, mode="energy-focused"), 12.0)
    assert "(max 12)" in energy and "11.70" in energy and "(+5.88)" in energy


def test_diverse_table_separates_relevance_and_adjusted_scores(catalog):
    picks = recommend_diverse_songs(ROCK, catalog)
    header, *rows = parse_table(diverse_table(picks, 11.0))
    assert column_text(header[3]) == "Relevance (max 11) -> Adjusted"
    static = rows[3]
    assert static[1][0] == "Static Postcards"
    assert static[3] == ["6.86", "-> 3.86"]
    assert ("Diversity: Genre rock already picked 2x (-3.00); "
            "adjusted 3.86 = relevance 6.86 minus penalties") in column_text(static[4])
    assert "Diversity: No repetition penalty" in column_text(rows[0][4])
    for row, pick in zip(rows, picks):
        assert row[3] == [f"{pick.base_score:.2f}", f"-> {pick.adjusted_score:.2f}"]
        for reason in pick.explanation.split("; "):
            assert reason in column_text(row[4])


def test_diverse_table_shows_artist_penalty():
    songs = [
        ({"id": 1, "title": "First", "artist": "Same Act", "genre": "pop", "mood": "happy"}, 9.0, "Genre match: pop (+4.00)"),
        ({"id": 2, "title": "Second", "artist": "Same Act", "genre": "jazz", "mood": "chill"}, 8.0, "Mood x (+0.00)"),
    ]
    _, _, second = parse_table(diverse_table(diversity_rerank(songs, k=2), 11.0))
    assert second[3] == ["8.00", "-> 6.00"]
    assert ("Diversity: Artist Same Act already picked (-2.00); "
            "adjusted 6.00 = relevance 8.00 minus penalties") in column_text(second[4])


def test_rendering_does_not_change_recommendations(catalog):
    recs = recommend_songs(POP, catalog)
    snapshot = copy.deepcopy(recs)
    recommendations_table(recs, 11.0)
    assert recs == snapshot
    picks = recommend_diverse_songs(POP, catalog)
    picks_snapshot = copy.deepcopy(picks)
    diverse_table(picks, 11.0)
    assert picks == picks_snapshot
    assert recommend_songs(POP, catalog) == snapshot


def test_reason_column_width_is_respected(catalog):
    text = recommendations_table(recommend_songs(POP, catalog), 11.0)
    for line in text.splitlines():
        if line.startswith("|"):
            assert len(line.rsplit(" | ", 1)[1]) == REASON_WIDTH + 2  # content + " |"


# ---------------------------------------------------------------- CLI

def run(capsys, *args):
    main(list(args))
    return capsys.readouterr().out


def test_default_output_is_unchanged_by_table_feature(capsys):
    default = run(capsys)
    assert "+----" not in default
    assert "1. Sunrise City by Neon Echo" in default


@pytest.mark.parametrize(
    "args, tables",
    [
        (["--table"], 1),
        (["--all-profiles", "--table"], 4),
        (["--mode", "mood-first", "--table"], 1),
        (["--diverse", "--table"], 1),
        (["--mode", "energy-focused", "--diverse", "--table"], 1),
    ],
)
def test_cli_table_combinations(capsys, args, tables):
    out = run(capsys, *args)
    assert sum(line.startswith("+=") for line in out.splitlines()) == tables  # one header rule per table
    assert "Because:" not in out  # table replaces the plain listing
    if "--diverse" in args:
        assert "-> Adjusted" in out and "Diversity reranking: on" in out
    if "mood-first" in args:
        assert "Ranking mode: mood-first" in out and "(max 10)" in out


def test_table_lists_the_same_songs_in_the_same_order_as_plain_output(capsys):
    for extra in ([], ["--diverse"], ["--mode", "energy-focused"]):
        plain = run(capsys, "--all-profiles", *extra)
        table = run(capsys, "--all-profiles", "--table", *extra)
        plain_titles = [line.split(". ", 1)[1].split(" by ")[0] for line in plain.splitlines()
                        if line[:1].isdigit() and ". " in line]
        table_titles = [line.split(" | ")[1].strip() for line in table.splitlines()
                        if line.startswith("| ") and line[2:4].strip().isdigit()]
        assert table_titles == plain_titles and len(plain_titles) == 20
