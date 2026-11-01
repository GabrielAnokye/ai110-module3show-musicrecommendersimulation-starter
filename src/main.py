"""
Command line runner for the Music Recommender Simulation.

Run from the project root with:  python -m src.main
"""

from pathlib import Path

from src.recommender import load_songs, recommend_songs

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


def main() -> None:
    """Load the catalog, score it for the default user, and print the top 5."""
    songs = load_songs(str(DATA_PATH))
    print(f"Loaded songs: {len(songs)}")

    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
    }

    print("\nUser preferences:")
    for key, value in user_prefs.items():
        print(f"  {key}: {value}")

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{rank}. {song['title']} by {song['artist']}")
        print(f"   Genre: {song['genre']} | Mood: {song['mood']} | Score: {score:.2f}")
        print(f"   Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
