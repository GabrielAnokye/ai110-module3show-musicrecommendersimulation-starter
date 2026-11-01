"""
ASCII table rendering for the CLI's --table option.

Formatting only: these functions take results already produced by recommend_songs or
recommend_diverse_songs and lay them out. They never score or rank anything.
"""

import textwrap
from typing import Dict, List, Sequence, Tuple

from src.recommender import REASON_SEPARATOR, DiversePick

# Column widths (characters of content, excluding the "| " borders). Fits a 100-column terminal.
RANK_WIDTH = 2
SONG_WIDTH = 20
LABEL_WIDTH = 10
SCORE_WIDTH = 11
REASON_WIDTH = 40
CONTINUATION_INDENT = "  "


def wrap_cell(text: str, width: int) -> List[str]:
    """
    Wrap one cell's text to the column width without dropping anything.

    Explicit newlines start new lines; each piece is wrapped on spaces, continuation lines
    are indented, and a word longer than the column is split rather than clipped.
    """
    lines: List[str] = []
    for piece in text.split("\n"):
        wrapped = textwrap.wrap(
            piece,
            width=width,
            subsequent_indent=CONTINUATION_INDENT,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[int]) -> str:
    """Render rows as a bordered, fixed-width ASCII table; cells wrap onto extra lines."""
    if not len(headers) == len(widths) or any(len(row) != len(widths) for row in rows):
        raise ValueError("every row and the headers must have one cell per column")
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_border = border.replace("-", "=")

    def render_row(cells: Sequence[str]) -> List[str]:
        wrapped = [wrap_cell(cell, width) for cell, width in zip(cells, widths)]
        height = max(len(lines) for lines in wrapped)
        return [
            "| " + " | ".join(
                (lines[i] if i < len(lines) else "").ljust(width)
                for lines, width in zip(wrapped, widths)
            ) + " |"
            for i in range(height)
        ]

    out = [border, *render_row(headers), header_border]
    for row in rows:
        out.extend(render_row(row))
        out.append(border)
    return "\n".join(out)


def _song_cell(song: Dict) -> str:
    return f"{song['title']}\nby {song['artist']}"


def _label_cell(song: Dict) -> str:
    return f"{song['genre']}\n{song['mood']}"


def _reasons_cell(explanation: str) -> str:
    # One reason per line, exactly as score_song wrote them.
    return "\n".join(explanation.split(REASON_SEPARATOR))


def _max_label(max_score: float) -> str:
    return f"(max {max_score:g})"


def recommendations_table(recommendations: List[Tuple[Dict, float, str]], max_score: float) -> str:
    """Table for recommend_songs output: (song, score, explanation) tuples."""
    headers = ["#", "Song / Artist", "Genre/Mood", f"Score\n{_max_label(max_score)}", "Why (score breakdown)"]
    rows = [
        [str(rank), _song_cell(song), _label_cell(song), f"{score:.2f}", _reasons_cell(explanation)]
        for rank, (song, score, explanation) in enumerate(recommendations, start=1)
    ]
    return render_table(headers, rows, [RANK_WIDTH, SONG_WIDTH, LABEL_WIDTH, SCORE_WIDTH, REASON_WIDTH])


def diverse_table(picks: List[DiversePick], max_score: float) -> str:
    """Table for recommend_diverse_songs output, keeping relevance and adjusted scores apart."""
    headers = [
        "#", "Song / Artist", "Genre/Mood",
        f"Relevance\n{_max_label(max_score)}\n-> Adjusted", "Why (score breakdown and\ndiversity penalty)",
    ]
    rows = []
    for rank, pick in enumerate(picks, start=1):
        # The whole penalty explanation stays one entry, so it reads as one diversity note.
        reasons = _reasons_cell(pick.explanation) + "\nDiversity: " + pick.penalty_explanation()
        rows.append([
            str(rank), _song_cell(pick.song), _label_cell(pick.song),
            f"{pick.base_score:.2f}\n-> {pick.adjusted_score:.2f}", reasons,
        ])
    return render_table(headers, rows, [RANK_WIDTH, SONG_WIDTH, LABEL_WIDTH, SCORE_WIDTH, REASON_WIDTH])
