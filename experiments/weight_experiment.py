"""
Phase 4 controlled weight experiment.

Compares the baseline weights (genre 4, mood 3, energy 3, acoustic 1; max 11)
with an experimental configuration (genre 2, mood 3, energy 6, acoustic 1; max 12)
on the same catalog, profiles, top-k, and ranking rules.

The experimental weights are applied with unittest.mock.patch.multiple, so the real
score_song / recommend_songs code is used and src/recommender.py is never edited.
The patch is undone automatically when the `with` block exits, even on an error.

Run from the project root:
    python -m experiments.weight_experiment
"""

from contextlib import contextmanager
from typing import Dict, List
from unittest.mock import patch

from src import recommender
from src.main import DATA_PATH, PROFILES, TOP_K

BASELINE = {"GENRE_WEIGHT": 4.0, "MOOD_WEIGHT": 3.0, "ENERGY_WEIGHT": 3.0, "ACOUSTIC_WEIGHT": 1.0}
EXPERIMENT = {"GENRE_WEIGHT": 2.0, "MOOD_WEIGHT": 3.0, "ENERGY_WEIGHT": 6.0, "ACOUSTIC_WEIGHT": 1.0}

# Two extra edge-case profiles (valid input, unusual preferences). Evaluated with the baseline only.
EXTRA_PROFILES: Dict[str, Dict] = {
    "Energetic Acoustic Folk": {
        "favorite_genre": "folk",
        "favorite_mood": "happy",
        "target_energy": 0.85,
        "likes_acoustic": True,
    },
    "Genre Not In Catalog (metal)": {
        "favorite_genre": "metal",
        "favorite_mood": "intense",
        "target_energy": 0.95,
        "likes_acoustic": False,
    },
}


def current_weights() -> Dict[str, float]:
    return {name: getattr(recommender, name) for name in BASELINE}


@contextmanager
def weights(config: Dict[str, float]):
    """Temporarily set the recommender's weight constants (and MAX_SCORE)."""
    max_score = sum(config.values())
    with patch.multiple(recommender, MAX_SCORE=max_score, **config):
        yield max_score


def full_ranking(profile: Dict, songs: List[Dict]):
    """Rank the whole catalog with the real implementation."""
    return recommender.recommend_songs(profile, songs, k=len(songs))


def describe(ranking, k=TOP_K) -> List[str]:
    lines = []
    for rank, (song, score, explanation) in enumerate(ranking[:k], start=1):
        lines.append(f"  {rank}. #{song['id']:<2} {song['title']:<20} ({song['genre']}, {song['mood']})  {score:.2f}")
        lines.append(f"       {explanation}")
    return lines


def main() -> None:
    songs = recommender.load_songs(str(DATA_PATH))
    assert current_weights() == BASELINE, f"unexpected starting weights: {current_weights()}"

    # Sanity check that the patch reaches score_song: Sunrise City for the pop profile.
    pop, sunrise = PROFILES["High-Energy Pop"], songs[0]
    base_score, _ = recommender.score_song(pop, sunrise)
    with weights(EXPERIMENT) as max_score:
        exp_score, exp_reasons = recommender.score_song(pop, sunrise)
        assert max_score == 12.0
    print("Patch check (Sunrise City, High-Energy Pop):")
    print(f"  baseline score     {base_score:.4f}")
    print(f"  experimental score {exp_score:.4f}  <- {'; '.join(exp_reasons)}")
    assert abs(base_score - exp_score) > 1e-9, "patch did not affect score_song"

    baseline_runs, experiment_runs = {}, {}
    for name, profile in PROFILES.items():
        baseline_runs[name] = full_ranking(profile, songs)
    with weights(EXPERIMENT):
        for name, profile in PROFILES.items():
            experiment_runs[name] = full_ranking(profile, songs)

    assert current_weights() == BASELINE, "baseline weights were not restored"
    assert recommender.MAX_SCORE == 11.0, "baseline MAX_SCORE was not restored"

    for name in PROFILES:
        base, exp = baseline_runs[name], experiment_runs[name]
        base_top = [s["id"] for s, _, _ in base[:TOP_K]]
        exp_top = [s["id"] for s, _, _ in exp[:TOP_K]]
        base_pos = {s["id"]: i for i, (s, _, _) in enumerate(base, start=1)}
        exp_pos = {s["id"]: i for i, (s, _, _) in enumerate(exp, start=1)}
        base_score = {s["id"]: sc for s, sc, _ in base}
        exp_score = {s["id"]: sc for s, sc, _ in exp}
        title = {s["id"]: s["title"] for s in songs}
        genre = {s["id"]: s["genre"] for s in songs}
        energy = {s["id"]: s["energy"] for s in songs}
        target = PROFILES[name]["target_energy"]

        print()
        print("=" * 100)
        print(f"{name}: {PROFILES[name]}")
        print("=" * 100)
        print("BASELINE (4/3/3/1, max 11):")
        print("\n".join(describe(base)))
        print("EXPERIMENT (2/3/6/1, max 12):")
        print("\n".join(describe(exp)))

        print(f"Top song: baseline #{base_top[0]} {title[base_top[0]]} | "
              f"experiment #{exp_top[0]} {title[exp_top[0]]} | "
              f"{'unchanged' if base_top[0] == exp_top[0] else 'CHANGED'}")
        kept = [i for i in base_top if i in exp_top]
        print(f"Songs kept in top {TOP_K}: {len(kept)}/{TOP_K}")
        print(f"Entered: {[f'#{i} {title[i]}' for i in exp_top if i not in base_top]}")
        print(f"Left:    {[f'#{i} {title[i]}' for i in base_top if i not in exp_top]}")
        print("Rank changes (union of both top-5 lists):")
        for song_id in sorted(set(base_top) | set(exp_top), key=lambda i: exp_pos[i]):
            delta = base_pos[song_id] - exp_pos[song_id]
            arrow = f"up {delta}" if delta > 0 else f"down {-delta}" if delta < 0 else "same"
            print(f"  #{song_id:<2} {title[song_id]:<20} {genre[song_id]:<10} energy {energy[song_id]:.2f} "
                  f"(|gap| {abs(energy[song_id] - target):.2f})  "
                  f"rank {base_pos[song_id]:>2} -> {exp_pos[song_id]:>2} ({arrow:<7})  "
                  f"score {base_score[song_id]:.2f} -> {exp_score[song_id]:.2f}")
        pref = PROFILES[name]["favorite_genre"]
        for label, top in (("baseline", base_top), ("experiment", exp_top)):
            cross = sum(1 for i in top if recommender.normalize_label(genre[i]) != pref)
            mean_gap = sum(abs(energy[i] - target) for i in top) / len(top)
            print(f"  {label:<10}: cross-genre songs in top {TOP_K} = {cross}, "
                  f"mean |energy - target| = {mean_gap:.3f}")

    print()
    print("=" * 100)
    print("EXTRA EDGE-CASE PROFILES (baseline weights only)")
    print("=" * 100)
    for name, profile in EXTRA_PROFILES.items():
        print(f"{name}: {profile}")
        print("\n".join(describe(full_ranking(profile, songs))))
        print()

    assert current_weights() == BASELINE
    print(f"Weights after experiment: {current_weights()}, MAX_SCORE={recommender.MAX_SCORE}")


if __name__ == "__main__":
    main()
