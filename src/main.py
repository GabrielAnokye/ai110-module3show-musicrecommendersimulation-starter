"""
Command line runner for the Music Recommender Simulation.

Run from the project root with:
    python -m src.main                  # default High-Energy Pop profile
    python -m src.main --all-profiles   # all four Phase 4 evaluation profiles
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from src.recommender import load_songs, recommend_songs

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
TOP_K = 5

# Hypothetical evaluation profiles (Phase 4). These are not real listener data.
PROFILES: Dict[str, Dict] = {
    "High-Energy Pop": {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.80,
        "likes_acoustic": False,
    },
    "Chill Lofi": {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.40,
        "likes_acoustic": True,
    },
    "Intense Rock": {
        "favorite_genre": "rock",
        "favorite_mood": "intense",
        "target_energy": 0.90,
        "likes_acoustic": False,
    },
    # Deliberately conflicting: a calm genre and a sad mood with a high energy target.
    "Adversarial Listener": {
        "favorite_genre": "lofi",
        "favorite_mood": "sad",
        "target_energy": 0.90,
        "likes_acoustic": False,
    },
}
DEFAULT_PROFILE = "High-Energy Pop"


def print_recommendations(user_prefs: Dict, songs: List[Dict], k: int = TOP_K) -> None:
    """Print a profile's preferences and its top-k recommendations with scores and reasons."""
    print("\nUser preferences:")
    for key, value in user_prefs.items():
        print(f"  {key}: {value}")

    recommendations = recommend_songs(user_prefs, songs, k=k)

    print("\nTop recommendations:\n")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{rank}. {song['title']} by {song['artist']}")
        print(f"   Genre: {song['genre']} | Mood: {song['mood']} | Score: {score:.2f}")
        print(f"   Because: {explanation}")
        print()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Music Recommender Simulation")
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="show recommendations for all four evaluation profiles",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Load the catalog and print the top 5 for the default profile, or for every profile."""
    args = parse_args(argv)
    songs = load_songs(str(DATA_PATH))
    print(f"Loaded songs: {len(songs)}")

    if not args.all_profiles:
        print_recommendations(PROFILES[DEFAULT_PROFILE], songs)
        return

    for name, user_prefs in PROFILES.items():
        print()
        print("=" * 70)
        print(f"Profile: {name}")
        print("=" * 70)
        print_recommendations(user_prefs, songs)


if __name__ == "__main__":
    main()
