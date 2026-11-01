import csv
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

# Scoring weights (finalized in Phase 2). Maximum possible score is 11.0.
GENRE_WEIGHT = 4.0
MOOD_WEIGHT = 3.0
ENERGY_WEIGHT = 3.0
ACOUSTIC_WEIGHT = 1.0
MAX_SCORE = GENRE_WEIGHT + MOOD_WEIGHT + ENERGY_WEIGHT + ACOUSTIC_WEIGHT

REQUIRED_COLUMNS = [
    "id", "title", "artist", "genre", "mood",
    "energy", "tempo_bpm", "valence", "danceability", "acousticness",
]
UNIT_RANGE_COLUMNS = ["energy", "valence", "danceability", "acousticness"]
REQUIRED_PREFERENCES = ["favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"]
REASON_SEPARATOR = "; "


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k Song objects, ranked by the shared functional recommender."""
        song_dicts = [asdict(song) for song in self.songs]
        # recommend_songs returns the same dict objects it was given, so map each back to its Song.
        song_for_dict = {id(song_dict): song for song_dict, song in zip(song_dicts, self.songs)}
        ranked = recommend_songs(asdict(user), song_dicts, k)
        return [song_for_dict[id(song_dict)] for song_dict, _, _ in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain a song's score using the same reasons score_song produces."""
        _, reasons = score_song(asdict(user), asdict(song))
        return REASON_SEPARATOR.join(reasons)


def normalize_label(text: str) -> str:
    """Trim, collapse repeated whitespace, and case-fold a genre or mood label."""
    return " ".join(text.split()).casefold()


def _parse_number(value: str, column: str, where: str) -> float:
    """Convert a CSV cell to a finite float, or raise ValueError naming the row."""
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{where}: {column} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{where}: {column} must be a finite number, got {value!r}")
    return number


def load_songs(csv_path: str) -> List[Dict]:
    """Load and validate the song catalog from a CSV file into a list of dicts."""
    try:
        csv_file = open(csv_path, newline="", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Song catalog not found: {csv_path}") from None

    songs: List[Dict] = []
    seen_ids = set()
    with csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: file is empty (no header row)")
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"{csv_path}: missing required column(s): {', '.join(missing_columns)}")

        for row in reader:
            where = f"{csv_path}, line {reader.line_num}"
            if None in row:  # DictReader stores extra cells under the key None
                raise ValueError(f"{where}: row has more values than the header")
            for col in REQUIRED_COLUMNS:
                if row[col] is None or row[col].strip() == "":
                    raise ValueError(f"{where}: missing value for {col}")

            try:
                song_id = int(row["id"])
            except ValueError:
                raise ValueError(f"{where}: id must be an integer, got {row['id']!r}") from None
            if song_id in seen_ids:
                raise ValueError(f"{where}: duplicate id {song_id}")
            seen_ids.add(song_id)

            song: Dict = {
                "id": song_id,
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
            }
            for col in ["energy", "tempo_bpm", "valence", "danceability", "acousticness"]:
                song[col] = _parse_number(row[col], col, where)
            for col in UNIT_RANGE_COLUMNS:
                if not 0.0 <= song[col] <= 1.0:
                    raise ValueError(f"{where}: {col} must be between 0 and 1, got {song[col]}")
            if song["tempo_bpm"] <= 0:
                raise ValueError(f"{where}: tempo_bpm must be positive, got {song['tempo_bpm']}")
            songs.append(song)
    return songs


def _is_number(value: object) -> bool:
    """True for finite ints/floats, excluding bools."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_user_prefs(user_prefs: Dict) -> None:
    """Raise ValueError unless the profile has all four valid preference fields."""
    if not isinstance(user_prefs, dict):
        raise ValueError("user_prefs must be a dict")
    missing = [key for key in REQUIRED_PREFERENCES if key not in user_prefs]
    if missing:
        raise ValueError(f"user_prefs is missing required field(s): {', '.join(missing)}")
    for key in ["favorite_genre", "favorite_mood"]:
        if not isinstance(user_prefs[key], str) or not user_prefs[key].strip():
            raise ValueError(f"{key} must be a non-empty string, got {user_prefs[key]!r}")
    target = user_prefs["target_energy"]
    if not _is_number(target) or not 0.0 <= target <= 1.0:
        raise ValueError(f"target_energy must be a number between 0 and 1, got {target!r}")
    if not isinstance(user_prefs["likes_acoustic"], bool):
        raise ValueError(f"likes_acoustic must be True or False, got {user_prefs['likes_acoustic']!r}")


def _validate_song(song: Dict) -> None:
    """Raise ValueError unless the song has the fields score_song needs."""
    for key in ["genre", "mood"]:
        if not isinstance(song.get(key), str):
            raise ValueError(f"song {key} must be a string, got {song.get(key)!r}")
    for key in ["energy", "acousticness"]:
        value = song.get(key)
        if not _is_number(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"song {key} must be a number between 0 and 1, got {value!r}")


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against a user profile; return (score, reasons)."""
    _validate_user_prefs(user_prefs)
    _validate_song(song)
    reasons: List[str] = []

    wanted_genre = " ".join(user_prefs["favorite_genre"].split())
    if normalize_label(song["genre"]) == normalize_label(wanted_genre):
        genre_points = GENRE_WEIGHT
        reasons.append(f"Genre match: {song['genre']} (+{genre_points:.2f})")
    else:
        genre_points = 0.0
        reasons.append(f"Genre {song['genre']} is not {wanted_genre} (+{genre_points:.2f})")

    wanted_mood = " ".join(user_prefs["favorite_mood"].split())
    if normalize_label(song["mood"]) == normalize_label(wanted_mood):
        mood_points = MOOD_WEIGHT
        reasons.append(f"Mood match: {song['mood']} (+{mood_points:.2f})")
    else:
        mood_points = 0.0
        reasons.append(f"Mood {song['mood']} is not {wanted_mood} (+{mood_points:.2f})")

    target = user_prefs["target_energy"]
    energy_similarity = max(0.0, 1.0 - abs(song["energy"] - target))
    energy_points = ENERGY_WEIGHT * energy_similarity
    reasons.append(f"Energy {song['energy']:.2f} vs target {target:.2f} (+{energy_points:.2f})")

    if user_prefs["likes_acoustic"]:
        acoustic_points = ACOUSTIC_WEIGHT * song["acousticness"]
        reasons.append(
            f"Acousticness {song['acousticness']:.2f}, you like acoustic (+{acoustic_points:.2f})"
        )
    else:
        acoustic_points = ACOUSTIC_WEIGHT * (1.0 - song["acousticness"])
        reasons.append(
            f"Acousticness {song['acousticness']:.2f}, you prefer less acoustic (+{acoustic_points:.2f})"
        )

    score = genre_points + mood_points + energy_points + acoustic_points
    return score, reasons


def _validate_k(k: int) -> None:
    """Raise TypeError for a non-integer k and ValueError for a negative k."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"k must be an integer, got {k!r}")
    if k < 0:
        raise ValueError(f"k must be 0 or greater, got {k}")


def _ranking_key(song: Dict, score: float) -> Tuple[float, int]:
    """Sort key: higher score first (rounded so float noise can't decide ties), then lower id."""
    song_id = song.get("id")
    if not isinstance(song_id, int) or isinstance(song_id, bool):
        raise ValueError(f"song id must be an integer, got {song_id!r}")
    return (-round(score, 6), song_id)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song, rank by score (ties: lower id), and return the top k as (song, score, explanation)."""
    _validate_k(k)
    _validate_user_prefs(user_prefs)

    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, REASON_SEPARATOR.join(reasons)))

    # sorted() returns a new list, so the caller's catalog keeps its original order.
    ranked = sorted(scored, key=lambda item: _ranking_key(item[0], item[1]))
    return ranked[:k]
